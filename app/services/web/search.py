import os
import requests
import logfire
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS


class WebSearchService:
    """
    Modular web search service supporting:
    1. Tavily Search API (primary, structured for AI agents).
    2. DuckDuckGo Search (automatic fallback when Tavily key is missing or rate limited).
    """

    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

    def search_tavily(self, query: str, max_results: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Search via Tavily REST API."""
        if not self.tavily_api_key:
            return None

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": max_results
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                        "score": float(item.get("score", 0.9)),
                        "source": "tavily"
                    })
                return results
            else:
                logfire.warning(f"Tavily API responded with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logfire.warning(f"Tavily search failed: {str(e)}")
            return None

    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search via DuckDuckGo as reliable fallback with query simplification retry."""
        clean_query = query.strip(' "\'')
        queries_to_try = [clean_query]
        words = clean_query.split()
        if len(words) > 6:
            queries_to_try.append(" ".join(words[:6]))

        for q in queries_to_try:
            try:
                results = []
                with DDGS() as ddgs:
                    ddg_results = list(ddgs.text(q, max_results=max_results))
                    for item in ddg_results:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                            "score": 0.8,
                            "source": "duckduckgo"
                        })
                if results:
                    return results
            except Exception as e:
                logfire.warning(f"DuckDuckGo search attempt failed for query '{q}': {str(e)}")

        return []

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Execute search across available providers (Tavily first, then DuckDuckGo).
        Returns a normalized dictionary.
        """
        with logfire.span("Web Search", query=query):
            # Attempt 1: Tavily
            tavily_results = self.search_tavily(query, max_results=max_results)
            if tavily_results is not None and len(tavily_results) > 0:
                return {
                    "success": True,
                    "query": query,
                    "provider": "tavily",
                    "results": tavily_results,
                    "count": len(tavily_results)
                }

            # Attempt 2: DuckDuckGo Fallback
            ddg_results = self.search_duckduckgo(query, max_results=max_results)
            if ddg_results:
                return {
                    "success": True,
                    "query": query,
                    "provider": "duckduckgo",
                    "results": ddg_results,
                    "count": len(ddg_results)
                }

            # If both failed or returned empty
            return {
                "success": False,
                "query": query,
                "provider": "none",
                "results": [],
                "count": 0,
                "error": "No external search results found or search services are currently unreachable."
            }


# Global search service instance
web_search_service = WebSearchService()

