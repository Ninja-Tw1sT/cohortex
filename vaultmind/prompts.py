"""Role-based prompt assembly. Adapts ai-workflow/agent/prompts.build_prompt into
a chat-message list (system + user with vault context + optional upstream input)."""
from __future__ import annotations

from vaultmind.profiles import AgentProfile

_CONTEXT_HEADER = "## Context from knowledge vault(s)\n\n"
_CHUNK = "### [{i}] {title} (distance {distance:.3f})\n{document}\n\n"


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
    if hits:
        user.append(_CONTEXT_HEADER)
        for i, h in enumerate(hits, 1):
            user.append(_CHUNK.format(
                i=i,
                title=h.get("title") or h.get("source") or "untitled",
                distance=h.get("distance", 0.0),
                document=(h.get("document") or "").strip(),
            ))
    if upstream:
        user.append(f"## Input from the previous step\n\n{upstream.strip()}\n\n")
    user.append(f"## Task\n\n{task.strip()}")
    return [
        {"role": "system", "content": build_system(profile)},
        {"role": "user", "content": "".join(user)},
    ]
