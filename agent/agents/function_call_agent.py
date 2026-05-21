from __future__ import annotations

from .prompt_agent import PromptAgentAdapter


class FunctionCallAgentAdapter(PromptAgentAdapter):
    """Placeholder adapter for tool-aware agents.

    The current outbound agent is prompt-only. This subclass keeps a stable name
    for future agents that emit structured tool calls while reusing the prompt
    adapter behavior today.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("agent_version", "function_call_prompt_agent")
        super().__init__(*args, **kwargs)
