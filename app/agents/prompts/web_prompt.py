"""Prompts for Web Search Agent."""

WEB_SEARCH_OPTIMIZER_PROMPT = """You are the Web Search Agent.
Your job is to formulate concise, targeted search queries to find the most accurate and recent external information.

User Query: {query}
Context/Goal: {context}

Generate 1 to 2 optimal keyword search queries.
Respond with valid JSON:
{{
  "search_queries": ["query 1", "query 2"]
}}
"""

WEB_SYNTHESIS_PROMPT = """You are the Web Search Agent.
Review the following web search results:

User Query: {query}

Search Results:
{results_formatted}

Instructions:
1. Extract relevant facts, statistics, dates, and conclusions from the search results.
2. Filter out irrelevant or promotional snippets.
3. Keep track of verified URLs for citations.
4. If search results are empty or irrelevant, report that external search could not find matching information.

Respond with valid JSON:
{{
  "summary": "Key facts extracted from the web...",
  "verified_sources": [
    {{"title": "Title", "url": "https://...", "snippet": "..."}}
  ],
  "confidence": 0.90
}}
"""
