import logfire
from typing import Dict, Any, List
from app.agents.state import AgentState
from app.services.web.search import web_search_service
from app.agents.llm import agent_llm
from app.agents.prompts.web_prompt import WEB_SEARCH_OPTIMIZER_PROMPT, WEB_SYNTHESIS_PROMPT


def web_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Web Search Agent.
    Searches external web sources via Tavily (with DuckDuckGo fallback),
    extracts factual snippets, preserves verified URLs, and handles failures cleanly.
    """
    query = state.get("user_query", "")
    with logfire.span("Web Agent Node", query=query):
        # 1. Optimize search query if needed
        optimizer_prompt = WEB_SEARCH_OPTIMIZER_PROMPT.format(
            query=query,
            context="Enterprise research query"
        )
        opt_data = agent_llm.call_json([
            {"role": "system", "content": "You are a web search keyword optimizer."},
            {"role": "user", "content": optimizer_prompt}
        ])
        
        search_queries = opt_data.get("search_queries", [query])
        primary_query = search_queries[0] if search_queries else query

        # 2. Execute Web Search
        search_res = web_search_service.search(primary_query, max_results=5)
        raw_results = search_res.get("results", [])

        if not raw_results:
            return {
                "web_search_results": [],
                "agent_messages": [{
                    "agent": "web_agent",
                    "action": "search",
                    "status": "warning",
                    "message": f"No web search results found for query '{primary_query}'."
                }],
                "errors": ["Web Agent: External search was unreachable or returned no results."]
            }

        # 3. Format and preserve verified source citations
        from urllib.parse import urlparse
        citations: List[Dict[str, Any]] = []
        formatted_snippets = []
        for i, item in enumerate(raw_results, 1):
            title = item.get("title", "Web Result")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            score = item.get("score", 0.8)
            domain = urlparse(url).netloc if url else ""

            citations.append({
                "type": "web",
                "title": title,
                "url": url,
                "domain": domain,
                "score": score,
                "snippet": snippet[:150] + "..." if len(snippet) > 150 else snippet
            })

            formatted_snippets.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}")

        snippets_text = "\n\n".join(formatted_snippets)

        # 4. Synthesize web findings
        synthesis_prompt = WEB_SYNTHESIS_PROMPT.format(
            query=query,
            results_formatted=snippets_text
        )
        synthesis = agent_llm.call_json([
            {"role": "system", "content": "You are a web information synthesis agent."},
            {"role": "user", "content": synthesis_prompt}
        ])
        web_summary = synthesis.get("summary", snippets_text[:1200])

        existing_context = state.get("final_context", "")
        updated_context = f"{existing_context}\n### External Web Findings\n{web_summary}\n"

        return {
            "web_search_results": raw_results,
            "source_references": citations,
            "final_context": updated_context,
            "agent_messages": [{
                "agent": "web_agent",
                "action": "search_and_synthesis",
                "status": "success",
                "provider": search_res.get("provider", "unknown"),
                "results_count": len(raw_results)
            }]
        }

