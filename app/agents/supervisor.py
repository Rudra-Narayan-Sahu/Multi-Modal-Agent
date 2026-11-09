import re
import logfire
from typing import Dict, Any, List
from app.agents.state import AgentState
from app.agents.llm import agent_llm
from app.agents.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_EVALUATION_PROMPT


class SupervisorAgent:
    """
    Master controller of the Multi-Agent System.
    Analyzes queries, coordinates execution planning, performs dynamic routing,
    and terminates cycles safely.
    """

    @staticmethod
    def _fast_intent_check(query: str) -> Dict[str, Any]:
        """Sub-millisecond heuristic check for obvious queries."""
        q_lower = query.strip().lower()

        # Conversational greetings
        if q_lower in {"hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you"}:
            return {
                "category": "CONVERSATIONAL",
                "reasoning": "Greeting or pleasantry",
                "selected_agents": ["response_agent"],
                "execution_plan": ["response_agent"],
                "needs_research_synthesis": False,
                "next_step": "response_agent"
            }

        # Pure mathematical calculation
        if re.search(r"^\s*(\d+[\d\s\+\-\*\/\^\(\)\.\%]+\d+)\s*$", query) or any(
            w in q_lower for w in ["calculate", "sqrt(", "what is 2 +", "what is 100 *"]
        ):
            return {
                "category": "CALCULATION_QUERY",
                "reasoning": "Arithmetic or mathematical formula",
                "selected_agents": ["tool_agent", "response_agent"],
                "execution_plan": ["tool_agent", "response_agent"],
                "needs_research_synthesis": False,
                "next_step": "tool_agent"
            }

        return {}

    @classmethod
    def plan_and_route(cls, state: AgentState) -> Dict[str, Any]:
        """Initial classification and plan formulation node."""
        query = state.get("user_query", "")
        current_iter = state.get("iteration_count", 0) + 1

        with logfire.span("Supervisor Planning", query=query, iteration=current_iter):
            # Check fast heuristics
            fast_plan = cls._fast_intent_check(query)
            if fast_plan:
                return {
                    "query_category": fast_plan["category"],
                    "execution_plan": fast_plan["execution_plan"],
                    "selected_agents": fast_plan["selected_agents"],
                    "iteration_count": current_iter,
                    "execution_status": "planned",
                    "next_step": fast_plan["next_step"],
                    "agent_messages": [{
                        "agent": "supervisor",
                        "action": "fast_heuristic_plan",
                        "category": fast_plan["category"],
                        "next_step": fast_plan["next_step"]
                    }]
                }

            # LLM-based Planning with short-term context & long-term memory
            user_prompt_parts = []
            if state.get("conversation_context"):
                user_prompt_parts.append(f"Recent Conversation History (Short-Term Context):\n{state['conversation_context']}")
            if state.get("relevant_memories"):
                from app.memory.retriever import memory_retriever
                mem_text = memory_retriever.format_memories_prompt(state["relevant_memories"])
                if mem_text:
                    user_prompt_parts.append(mem_text)
            user_prompt_parts.append(f"Current User Query: {query}")
            full_user_prompt = "\n\n".join(user_prompt_parts)

            plan_data = agent_llm.call_json([
                {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": full_user_prompt}
            ])

            category = plan_data.get("category", "DOCUMENT_QUERY")
            selected_agents = plan_data.get("selected_agents", ["rag_agent"])
            execution_plan = plan_data.get("execution_plan", ["rag_agent", "response_agent"])
            needs_research = plan_data.get("needs_research_synthesis", False)

            # Determine immediate next step
            if category in {"COMPARISON_QUERY", "RESEARCH_QUERY"} or ("rag_agent" in selected_agents and "web_agent" in selected_agents):
                next_step = "parallel_rag_web"
            elif "rag_agent" in selected_agents:
                next_step = "rag_agent"
            elif "web_agent" in selected_agents:
                next_step = "web_agent"
            elif "tool_agent" in selected_agents:
                next_step = "tool_agent"
            else:
                next_step = "response_agent"

            return {
                "query_category": category,
                "execution_plan": execution_plan,
                "selected_agents": selected_agents,
                "iteration_count": current_iter,
                "execution_status": "planned",
                "next_step": next_step,
                "agent_messages": [{
                    "agent": "supervisor",
                    "action": "plan_and_route",
                    "category": category,
                    "selected_agents": selected_agents,
                    "next_step": next_step,
                    "reasoning": plan_data.get("reasoning", "")
                }]
            }

    @classmethod
    def evaluate_progress(cls, state: AgentState) -> Dict[str, Any]:
        """Intermediate validation node: decides whether to synthesize research or terminate."""
        query = state.get("user_query", "")
        current_iter = state.get("iteration_count", 0) + 1
        docs = state.get("retrieved_documents", [])
        web = state.get("web_search_results", [])
        category = state.get("query_category", "")
        research_done = state.get("research_findings") is not None

        with logfire.span("Supervisor Evaluation", query=query, iteration=current_iter):
            # Guard against runaway loops
            if current_iter >= 5:
                logfire.warning(f"Supervisor hit max iterations ({current_iter}). Forcing response node.")
                return {
                    "iteration_count": current_iter,
                    "next_step": "response_agent",
                    "agent_messages": [{
                        "agent": "supervisor",
                        "action": "terminate_cycle",
                        "reason": "Max iterations reached"
                    }]
                }

            # Check if research synthesis is needed
            if (docs and web) and not research_done:
                return {
                    "iteration_count": current_iter,
                    "next_step": "research_agent",
                    "agent_messages": [{
                        "agent": "supervisor",
                        "action": "route_to_research",
                        "reason": "Both documents and web results collected; activating comparative research."
                    }]
                }

            if category in {"COMPARISON_QUERY", "RESEARCH_QUERY"} and not research_done:
                return {
                    "iteration_count": current_iter,
                    "next_step": "research_agent",
                    "agent_messages": [{
                        "agent": "supervisor",
                        "action": "route_to_research",
                        "reason": f"Category is {category}; comparative synthesis required."
                    }]
                }

            # All intermediate steps complete -> forward to Response Agent
            return {
                "iteration_count": current_iter,
                "next_step": "response_agent",
                "agent_messages": [{
                    "agent": "supervisor",
                    "action": "route_to_response",
                    "reason": "Sufficient context gathered. Ready for final synthesis."
                }]
            }


supervisor_agent = SupervisorAgent()

