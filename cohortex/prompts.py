"""Role-based prompt assembly. Adapts ai-workflow/agent/prompts.build_prompt into
a chat-message list (system + user with vault context + optional upstream input)."""
from __future__ import annotations

from cohortex.profiles import AgentProfile

_CONTEXT_HEADER = "## Context from knowledge vault(s) — retrieved, top-k\n\n"
_CHUNK = "### [{i}] {title} (distance {distance:.3f})\n{document}\n\n"

_FULL_CONTEXT_HEADER = "## Full source document(s) — entire file loaded, no retrieval\n\n"
_FULL_CHUNK = "### [{i}] {title}\n{document}\n\n"


def build_system(profile: AgentProfile) -> str:
    parts = []
    if profile.role:
        parts.append(f"You are the {profile.role}.")
    if profile.goal:
        parts.append(f"Your goal: {profile.goal}.")
    if profile.system_prompt:
        parts.append(profile.system_prompt.strip())
    parts.append("Be concise and do only your job — do not add commentary.")
    return " ".join(parts) if len(parts) <= 2 else "\n\n".join(parts)


def build_messages(profile: AgentProfile, task: str,
                   hits: list[dict] | None = None, upstream: str = "") -> list[dict]:
    user: list[str] = []
    retrieved = [h for h in (hits or []) if not h.get("full_context")]
    full_docs = [h for h in (hits or []) if h.get("full_context")]
    if retrieved:
        user.append(_CONTEXT_HEADER)
        for i, h in enumerate(retrieved, 1):
            user.append(_CHUNK.format(
                i=i,
                title=h.get("title") or h.get("source") or "untitled",
                distance=h.get("distance", 0.0),
                document=(h.get("document") or "").strip(),
            ))
    if full_docs:
        user.append(_FULL_CONTEXT_HEADER)
        for i, h in enumerate(full_docs, 1):
            user.append(_FULL_CHUNK.format(
                i=i,
                title=h.get("title") or h.get("source") or "untitled",
                document=(h.get("document") or "").strip(),
            ))
    if upstream:
        user.append(f"## Input from the previous step\n\n{upstream.strip()}\n\n")
    user.append(f"## Task\n\n{task.strip()}")
    return [
        {"role": "system", "content": build_system(profile)},
        {"role": "user", "content": "".join(user)},
    ]
