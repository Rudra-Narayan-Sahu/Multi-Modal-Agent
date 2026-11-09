import logfire
from typing import Dict, Any, List
from app.agents.state import AgentState
from app.agents.llm import agent_llm
from app.agents.prompts.response_prompt import RESPONSE_AGENT_PROMPT


def response_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Response Agent.
    Synthesizes the complete multi-agent context into a coherent,
    structured markdown answer, appends verified citations,
    and documents limitations.
    """
    query = state.get("user_query", "")
    final_context = state.get("final_context", "")
    citations = state.get("source_references", [])
    errors = state.get("errors", [])

    with logfire.span("Response Agent Node", query=query):
        # Extract individual context sections
        doc_context = "\n".join([
            f"- [{d.get('file_name', 'Doc')} #{d.get('chunk_index', 0)}]: {d.get('text', '')[:400]}"
            for d in state.get("retrieved_documents", [])
        ]) or "None"

        web_context = "\n".join([
            f"- [{w.get('title', 'Web')}]: {w.get('snippet', '')} (URL: {w.get('url', '')})"
            for w in state.get("web_search_results", [])
        ]) or "None"

        research_context = state.get("research_findings") or "None"

        tool_context = "\n".join([
            f"- Tool '{t.get('tool_name')}': {t.get('result') or t.get('error')}"
            for t in state.get("tool_results", [])
        ]) or "None"

        prompt = RESPONSE_AGENT_PROMPT.format(
            query=query,
            doc_context=doc_context,
            web_context=web_context,
            research_context=research_context,
            tool_context=tool_context
        )

        response_body = agent_llm.call([
            {"role": "system", "content": "You are a professional research response generator."},
            {"role": "user", "content": prompt}
        ], temperature=0.3)

        # Build clean citations block
        doc_citations = [c for c in citations if c.get("type") == "document"]
        web_citations = [c for c in citations if c.get("type") == "web"]

        citation_sections: List[str] = []

        if doc_citations:
            doc_lines = ["\n### 📄 Document Sources"]
            seen_docs = set()
            for dc in doc_citations:
                key = (dc.get("file_name"), dc.get("chunk_index"))
                if key not in seen_docs:
                    seen_docs.add(key)
                    doc_lines.append(
                        f"- **{dc.get('file_name')}** (Chunk #{dc.get('chunk_index')}, Relevance: {dc.get('score')})"
                    )
            citation_sections.append("\n".join(doc_lines))

        if web_citations:
            web_lines = ["\n### 🌐 External Web Sources"]
            seen_urls = set()
            for wc in web_citations:
                url = wc.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    web_lines.append(f"- [{wc.get('title', 'Link')}]({url})")
            citation_sections.append("\n".join(web_lines))

        # Build limitations / warnings section if any errors occurred
        limitation_section = ""
        if errors:
            warning_lines = ["\n> [!NOTE]\n> **Execution Notice:**"]
            for err in errors:
                warning_lines.append(f"> - {err}")
            limitation_section = "\n" + "\n".join(warning_lines)

        full_final_markdown = response_body + "\n" + "\n".join(citation_sections) + limitation_section

        return {
            "final_response": full_final_markdown,
            "execution_status": "completed",
            "agent_messages": [{
                "agent": "response_agent",
                "action": "synthesize_final_response",
                "status": "success",
                "has_citations": len(citations) > 0
            }]
        }

