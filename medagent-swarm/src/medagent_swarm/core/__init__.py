"""Agent runtime public API, with deferred optional client dependencies."""
from .state_manager import AgentState, TaskStatus
from .skill_registry import SkillRegistry, SkillParameter

__all__ = ["LLMClient", "ToolCall", "LLMResponse", "AgentLoop", "AgentState", "TaskStatus", "SkillRegistry", "SkillParameter"]

def __getattr__(name):
    if name in ("LLMClient", "ToolCall", "LLMResponse"):
        from .llm_client import LLMClient, ToolCall, LLMResponse
        return {"LLMClient": LLMClient, "ToolCall": ToolCall, "LLMResponse": LLMResponse}[name]
    if name == "AgentLoop":
        from .agent_loop import AgentLoop
        return AgentLoop
    raise AttributeError(name)
