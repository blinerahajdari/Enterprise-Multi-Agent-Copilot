from __future__ import annotations 

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone


class AgentLogEntry(BaseModel):
    # Single trace entry for agent actions (who did what, when, and with what outcome)
    timestamp: str
    agent: str
    action: str
    outcome: str

    @staticmethod
    def now(agent: str, action: str, outcome: str) -> "AgentLogEntry":
        # Convenience constructor that stamps the log entry with current UTC time
        return AgentLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            action=action,
            outcome=outcome,
        )


class Citation(BaseModel):
    # Source reference attached to facts (doc identity, location, and supporting snippet)
    doc_id: str
    location: str
    snippet: str


class ResearchFact(BaseModel):
    # A single grounded fact plus its supporting citations
    fact: str
    citations: List[Citation]


class ResearchNotes(BaseModel):
    # Research outcome: either evidence was found ("ok") or not present in sources
    status: Literal["ok", "Not found in sources"]
    facts: List[ResearchFact] = Field(default_factory=list)


class AppState(BaseModel):
    # Inputs
    user_task: str = ""

    # Orchestration artifacts (planner output)
    plan: List[str] = Field(default_factory=list)

    # Research artifacts (researcher output)
    research_notes: Optional[ResearchNotes] = None

    # Writing artifacts (writer output)
    draft_output: Optional[str] = None

    # Final artifacts (accepted deliverable after verification)
    final_output: Optional[str] = None

    # Flattened citations for display/reporting (optional convenience)
    citations: List[Citation] = Field(default_factory=list)

    # Traceability: chronological agent logs
    agent_logs: List[AgentLogEntry] = Field(default_factory=list)

    # Controls for verifier retry/reroute loop
    verifier_fail_count: int = 0
    verifier_max_retries: int = 2

    # Extra metadata (model name, persist_dir, observability, etc.)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def log(self, agent: str, action: str, outcome: str) -> None:
        # Append a standardized log entry to the state
        self.agent_logs.append(AgentLogEntry.now(agent, action, outcome))
