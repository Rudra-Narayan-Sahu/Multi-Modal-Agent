import re
import logfire
from typing import Dict, Any
from app.agents.state import AgentState
from app.services.tools import tool_registry
from app.agents.llm import agent_llm
from app.agents.prompts.tool_prompt import TOOL_DISPATCH_PROMPT


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Tool Agent.
    Selects registered utility (Calculator, Python Executor, DateTime),
    safely executes it with argument validation, and records output.
    """
    query = state.get("user_query", "")
    with logfire.span("Tool Agent Node", query=query):
        # 1. Check for simple mathematical expression or direct calculation
        math_match = re.search(r"(\d+[\d\s\+\-\*\/\^\(\)\.\%]+\d+)", query)
        
        # 2. Use LLM to dispatch tool
        prompt = TOOL_DISPATCH_PROMPT.format(query=query)
        tool_call = agent_llm.call_json([
            {"role": "system", "content": "You are a tool dispatch specialist."},
            {"role": "user", "content": prompt}
        ])

        tool_name = tool_call.get("tool_name", "calculator")
        arguments = tool_call.get("arguments", {})

        # Heuristic fallback if LLM gave empty arguments
        if not arguments and math_match:
            tool_name = "calculator"
            arguments = {"expression": math_match.group(1)}

        # 3. Execute tool
        exec_result = tool_registry.execute(tool_name, arguments)

        tool_record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "success": exec_result.get("success", False),
            "result": exec_result.get("result", {}),
            "error": exec_result.get("error", None)
        }

        # Format context string
        if exec_result.get("success"):
            formatted_res = f"### Tool Execution Result ({tool_name})\n- Output: {exec_result.get('result')}\n"
        else:
            formatted_res = f"### Tool Execution Warning ({tool_name})\n- Error: {exec_result.get('error')}\n"

        existing_context = state.get("final_context", "")
        updated_context = f"{existing_context}\n{formatted_res}"

        return {
            "tool_results": [tool_record],
            "final_context": updated_context,
            "agent_messages": [{
                "agent": "tool_agent",
                "action": "execute_tool",
                "tool": tool_name,
                "success": exec_result.get("success", False)
            }]
        }

