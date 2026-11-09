from app.agents.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_EVALUATION_PROMPT
from app.agents.prompts.rag_prompt import RAG_ANALYSIS_PROMPT
from app.agents.prompts.web_prompt import WEB_SEARCH_OPTIMIZER_PROMPT, WEB_SYNTHESIS_PROMPT
from app.agents.prompts.research_prompt import RESEARCH_SYNTHESIS_PROMPT
from app.agents.prompts.tool_prompt import TOOL_DISPATCH_PROMPT
from app.agents.prompts.response_prompt import RESPONSE_AGENT_PROMPT

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "SUPERVISOR_EVALUATION_PROMPT",
    "RAG_ANALYSIS_PROMPT",
    "WEB_SEARCH_OPTIMIZER_PROMPT",
    "WEB_SYNTHESIS_PROMPT",
    "RESEARCH_SYNTHESIS_PROMPT",
    "TOOL_DISPATCH_PROMPT",
    "RESPONSE_AGENT_PROMPT",
]

