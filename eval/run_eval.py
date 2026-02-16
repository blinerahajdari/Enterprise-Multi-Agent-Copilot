import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()  # Initialize environment variables (API keys, config, etc.)


# Ensure project root is available in Python path when running this file directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.graph import run_task
from schemas.state import AppState


TEST_FILE = Path(__file__).parent / "test_cases.json"  # Location of test case definitions
CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"  # Persistent vector database directory


# ----------------------------
# Utility Functions
# ----------------------------
def normalize_state(result: Any) -> AppState:
    # Convert execution result into AppState (handles dict or model instance)
    if isinstance(result, AppState):
        return result
    if isinstance(result, dict):
        return AppState(**result)
    raise TypeError(f"Unexpected state type: {type(result)}")


def word_count(text: str) -> int:
    # Basic tokenizer-based word counting (used for length validation)
    return len(re.findall(r"\w+", text))


def contains_any(text: str, phrases: List[str]) -> bool:
    # Returns True if at least one phrase exists in the text (case-insensitive)
    lower = text.lower()
    return any(p.lower() in lower for p in phrases)


def contains_all(text: str, phrases: List[str]) -> bool:
    # Returns True only if all phrases exist in the text (case-insensitive)
    lower = text.lower()
    return all(p.lower() in lower for p in phrases)


# ----------------------------
# Test Execution Logic
# ----------------------------
def evaluate_test(test: Dict) -> Dict:
    # Execute a single test scenario and evaluate output against rules
    task = test["task"]
    checks = test.get("checks", {})

    print(f"\n--- Running {test['id']} ---")
    print(f"Task: {task}")

    result = run_task(user_task=task, persist_dir=str(CHROMA_DIR))
    state = normalize_state(result)

    # Use verified output if available, otherwise fallback to draft
    output = state.final_output or state.draft_output or ""
    output_lower = output.lower()

    failures = []

    # Required phrases validation
    for phrase in checks.get("must_include", []):
        if phrase.lower() not in output_lower:
            failures.append(f"Missing required phrase: '{phrase}'")

    # Forbidden phrases validation
    for phrase in checks.get("must_not_include", []):
        if phrase.lower() in output_lower:
            failures.append(f"Contains forbidden phrase: '{phrase}'")

    # At least one phrase from the provided list must appear
    if "must_include_any" in checks:
        if not contains_any(output, checks["must_include_any"]):
            failures.append(
                f"Must include at least one of: {checks['must_include_any']}"
            )

    # Word limit enforcement
    if "max_words" in checks:
        wc = word_count(output)
        if wc > checks["max_words"]:
            failures.append(
                f"Word count exceeded: {wc} > {checks['max_words']}"
            )

    # Explicit "not found" behavior enforcement when required
    if checks.get("must_return_not_found"):
        if "not found in sources" not in output_lower and "not documented" not in output_lower:
            failures.append("Expected 'Not found in sources' behavior.")

    passed = len(failures) == 0

    return {
        "id": test["id"],
        "passed": passed,
        "failures": failures,
        "output_preview": output[:300],  # Truncated preview for easier debugging
    }


# ----------------------------
# Entry Point
# ----------------------------
def main():
    # Load test configuration and execute all scenarios sequentially
    if not TEST_FILE.exists():
        print("test_cases.json not found.")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        tests = json.load(f)

    results = []
    for test in tests:
        res = evaluate_test(test)
        results.append(res)

        if res["passed"]:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            for fail in res["failures"]:
                print("   -", fail)

            print("\n----- MODEL OUTPUT -----")
            print(res["output_preview"])
            print("------------------------\n")

    total = len(results)
    passed = sum(r["passed"] for r in results)

    print("\n==============================")
    print(f"FINAL SCORE: {passed}/{total} passed")
    print("==============================")

    # Exit with non-zero status if any test fails (useful for CI/CD pipelines)
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
