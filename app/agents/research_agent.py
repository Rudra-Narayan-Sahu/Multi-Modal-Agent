import logfire
from typing import Dict, Any
from app.agents.state import AgentState
from app.agents.llm import agent_llm
from app.agents.prompts.research_prompt import RESEARCH_SYNTHESIS_PROMPT


def research_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Research Agent.
    Conducts deep comparative analysis between internal documents,
    live web search results, and tool outputs. Identifies contradictions,
    agreements, and research gaps.
    """
    query = state.get("user_query", "")
    docs = state.get("retrieved_documents", [])
    web_results = state.get("web_search_results", [])
    tool_results = state.get("tool_results", [])

    with logfire.span("Research Agent Node", query=query):
        # Format document findings
        if docs:
            doc_lines = [
                f"- [{d.get('file_name', 'Doc')} Chunk #{d.get('chunk_index', 0)}]: {d.get('text', '')[:300]}..."
                for d in docs[:4]
            ]
            doc_findings = "\n".join(doc_lines)
        else:
            doc_findings = "No internal document evidence available."

        # Format web findings
        if web_results:
            web_lines = [
                f"- [{w.get('title', 'Web')}]: {w.get('snippet', '')[:300]}... (URL: {w.get('url', '')})"
                for w in web_results[:4]
            ]
            web_findings = "\n".join(web_lines)
        else:
            web_findings = "No external web findings available."

        # Format tool outputs
        if tool_results:
            tool_findings = "\n".join([str(t) for t in tool_results])
        else:
            tool_findings = "No tool outputs."

        # Generate comparative research analysis
        prompt = RESEARCH_SYNTHESIS_PROMPT.format(
            query=query,
            doc_findings=doc_findings,
            web_findings=web_findings,
            tool_findings=tool_findings
        )

        research_analysis = agent_llm.call([
            {"role": "system", "content": "You are a senior research analyst."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)

        existing_context = state.get("final_context", "")
        updated_context = f"{existing_context}\n### Comparative Research Findings\n{research_analysis}\n"

        return {
            "research_findings": research_analysis,
            "final_context": updated_context,
            "agent_messages": [{
                "agent": "research_agent",
                "action": "comparative_synthesis",
                "status": "success",
                "docs_analyzed": len(docs),
                "web_sources_analyzed": len(web_results)
            }]
        }

