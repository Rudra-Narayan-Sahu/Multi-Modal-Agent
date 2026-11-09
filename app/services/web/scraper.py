import requests
from bs4 import BeautifulSoup
import logfire
from typing import Dict, Any


class WebScraper:
    """Safely extracts main body text from external web pages."""

    @staticmethod
    def scrape_url(url: str, timeout: int = 8, max_chars: int = 3000) -> Dict[str, Any]:
        """Fetch and extract readable text from a URL."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        with logfire.span("Web Scraping", url=url):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                if response.status_code != 200:
                    return {
                        "success": False,
                        "url": url,
                        "error": f"HTTP {response.status_code}"
                    }

                soup = BeautifulSoup(response.text, "html.parser")

                # Remove scripts, styles, and navigational elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.extract()

                text = soup.get_text(separator=" ", strip=True)
                clean_text = " ".join(text.split())

                if len(clean_text) > max_chars:
                    clean_text = clean_text[:max_chars] + "... [Content Truncated]"

                return {
                    "success": True,
                    "url": url,
                    "title": soup.title.string.strip() if soup.title and soup.title.string else "Untitled",
                    "content": clean_text
                }
            except Exception as e:
                return {
                    "success": False,
                    "url": url,
                    "error": f"ScrapeError: {str(e)}"
                }

