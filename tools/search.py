import requests
import json
from typing import Dict, List, Any
from datetime import datetime


class WebSearchTool:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def search(self, query: str, max_results: int = None) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "x-ai/grok-4-fast:online",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Search the web for: {query}. Provide factual, recent information with sources."
                        }
                    ],
                    "max_tokens": 10000,
                }
            )

            if response.status_code != 200:
                return {
                    "error": f"Search failed with status {response.status_code}",
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "results": []
                }

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            annotations = data["choices"][0]["message"].get("annotations", [])

            return {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "content": content,
                "results": self._parse_search_results(annotations, max_results)
            }

        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results": []
            }

    def _parse_search_results(self, annotations: List[Dict], max_results: int = None) -> List[Dict]:
        results = []

        for annotation in annotations:
            if annotation.get("type") == "url_citation":
                citation = annotation["url_citation"]
                results.append({
                    "title": citation.get("title", "Untitled"),
                    "url": citation["url"],
                    "content": citation.get("content", ""),
                    "type": "citation"
                })

        if max_results:
            results = results[:max_results]

        return results


def get_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information to verify claims. Use this when you need current data, statistics, or evidence from reliable sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and include relevant keywords for accurate results."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (optional, no limit)",
                        "minimum": 1
                    }
                },
                "required": ["query"]
            }
        }
    }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if api_key:
        tool = WebSearchTool(api_key)
        result = tool.search("Tesla FSD safety statistics 2024")
        print(json.dumps(result, indent=2))
    else:
        print("OPENROUTER_API_KEY not found in environment variables")


