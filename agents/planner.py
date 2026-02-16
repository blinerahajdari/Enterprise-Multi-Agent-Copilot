from __future__ import annotations  

import time

from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from schemas.state import AppState


class PlanOut(BaseModel):
    # Structured output schema for the planner
    steps: List[str] = Field(..., description="Ordered steps to complete the task.")


# System instructions that strictly define the planner's role and constraints
SYSTEM = """You are acting as the Planning Agent (Supply Chain Program Lead).

Your responsibility is to produce a clear and concise execution plan only.
Do NOT perform research.
Do NOT write or draft the final deliverable.

Guidelines:
- Generate 4 to 6 ordered steps.
- The steps must align with this flow: Plan → Research → Draft → Verify → Deliver.
- Indicate what type of supporting evidence should be gathered (KPIs, constraints, costs, risks).
- The response MUST be valid JSON that follows the defined schema.
"""

# Prompt template combining system instructions and user task
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "User task:\n{user_task}\n\nCreate the plan JSON now."),
    ]
)


def run_planner(state: AppState) -> AppState:
    t0 = time.time()  # Start latency measurement

    # Initialize LLM with deterministic output (temperature=0)
    llm = ChatOpenAI(model=state.meta.get("model", "gpt-4o-mini"), temperature=0)
    structured = llm.with_structured_output(PlanOut)  # Enforce structured JSON output

    # Invoke the model with formatted prompt messages
    out: PlanOut = structured.invoke(
        PROMPT.format_messages(user_task=state.user_task)
    )

    state.plan = out.steps  # Save generated plan into state
    state.log("planner", "created plan", f"{len(out.steps)} steps")

    # Observability: track execution latency and errors
    obs = state.meta.setdefault('observability', [])
    obs.append({
        'agent': 'planner',
        'latency_s': round(time.time() - t0, 3),
        'error': None,
    })

    return state
