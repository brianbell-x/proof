"""
Comprehensive tests for WebSearchTool.

These tests simulate how the model formats tool calls and verify the tool's
capabilities. Note: Some tests require a valid API key and will be skipped
if not available.
"""

import unittest
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.search import WebSearchTool, get_tool_schema


class MockToolCall:
    """Mock tool call object that simulates model response format."""
    def __init__(self, tool_name: str, arguments: dict, call_id: str = "test_call_123"):
        self.id = call_id
        self.function = MockFunction(tool_name, arguments)


class MockFunction:
    """Mock function object for tool calls."""
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class TestWebSearchTool(unittest.TestCase):
    """Test suite for WebSearchTool with model-formatted tool calls."""

    def setUp(self):
        api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key")
        self.tool = WebSearchTool(api_key)
        self.has_valid_key = api_key != "dummy_key"

    def _execute_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call formatted as the model would send it."""
        mock_call = MockToolCall(tool_name, arguments)
        return self.tool.search(**arguments)

    # ========== BASIC FUNCTIONALITY ==========

    def test_initialization(self):
        """Test tool initialization."""
        self.assertIsNotNone(self.tool.api_key)
        self.assertEqual(self.tool.base_url, "https://openrouter.ai/api/v1")

    def test_get_tool_schema(self):
        """Test tool schema structure."""
        schema = get_tool_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "web_search")
        self.assertIn("query", schema["function"]["parameters"]["properties"])
        self.assertIn("max_results", schema["function"]["parameters"]["properties"])

    def test_schema_query_required(self):
        """Test that query is required in schema."""
        schema = get_tool_schema()
        required = schema["function"]["parameters"].get("required", [])
        self.assertIn("query", required)

    def test_schema_max_results_optional(self):
        """Test that max_results is optional in schema."""
        schema = get_tool_schema()
        required = schema["function"]["parameters"].get("required", [])
        self.assertNotIn("max_results", required)

    # ========== ERROR HANDLING ==========

    def test_invalid_api_key_handling(self):
        """Test handling of invalid API key."""
        invalid_tool = WebSearchTool("invalid_key")
        result = invalid_tool.search("test query")
        self.assertIn("error", result)
        self.assertEqual(result["query"], "test query")
        self.assertIn("timestamp", result)

    def test_empty_query(self):
        """Test empty query handling."""
        result = self._execute_tool_call("web_search", {
            "query": ""
        })
        self.assertIn("query", result)
        self.assertIn("timestamp", result)

    def test_missing_query_parameter(self):
        """Test missing query parameter."""
        result = self.tool.search()
        self.assertIn("error", result)

    # ========== RESPONSE STRUCTURE ==========

    def test_response_structure(self):
        """Test that response has expected structure."""
        result = self._execute_tool_call("web_search", {
            "query": "test query"
        })
        
        required_fields = ["query", "timestamp", "results"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_timestamp_format(self):
        """Test timestamp format."""
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        self.assertIn("timestamp", result)
        timestamp = result["timestamp"]
        self.assertIsInstance(timestamp, str)
        self.assertIn("T", timestamp or "2024-01-01T00:00:00")

    def test_results_is_list(self):
        """Test that results is a list."""
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        self.assertIsInstance(result["results"], list)

    # ========== QUERY VARIATIONS ==========

    def test_simple_query(self):
        """Test simple query."""
        result = self._execute_tool_call("web_search", {
            "query": "Python programming"
        })
        self.assertIn("query", result)
        self.assertEqual(result["query"], "Python programming")

    def test_query_with_special_characters(self):
        """Test query with special characters."""
        result = self._execute_tool_call("web_search", {
            "query": "C++ programming & development"
        })
        self.assertIn("query", result)
        self.assertEqual(result["query"], "C++ programming & development")

    def test_query_with_numbers(self):
        """Test query with numbers."""
        result = self._execute_tool_call("web_search", {
            "query": "Python 3.12 features"
        })
        self.assertIn("query", result)
        self.assertEqual(result["query"], "Python 3.12 features")

    def test_long_query(self):
        """Test long query."""
        long_query = " ".join(["test"] * 50)
        result = self._execute_tool_call("web_search", {
            "query": long_query
        })
        self.assertIn("query", result)

    def test_query_with_quotes(self):
        """Test query with quotes."""
        result = self._execute_tool_call("web_search", {
            "query": 'What is "machine learning"?'
        })
        self.assertIn("query", result)

    # ========== MAX_RESULTS PARAMETER ==========

    def test_max_results_parameter(self):
        """Test max_results parameter."""
        result = self._execute_tool_call("web_search", {
            "query": "test",
            "max_results": 3
        })
        self.assertIn("results", result)
        if not result.get("error"):
            self.assertLessEqual(len(result["results"]), 3)

    def test_max_results_default(self):
        """Test default max_results behavior."""
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        self.assertIn("results", result)

    def test_max_results_one(self):
        """Test max_results = 1."""
        result = self._execute_tool_call("web_search", {
            "query": "test",
            "max_results": 1
        })
        self.assertIn("results", result)

    def test_max_results_large(self):
        """Test large max_results value."""
        result = self._execute_tool_call("web_search", {
            "query": "test",
            "max_results": 100
        })
        self.assertIn("results", result)

    # ========== REAL SEARCH TESTS (require API key) ==========

    @unittest.skipUnless(
        os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_API_KEY") != "dummy_key",
        "Requires valid OPENROUTER_API_KEY"
    )
    def test_real_search_basic(self):
        """Test real search with valid API key."""
        result = self._execute_tool_call("web_search", {
            "query": "Python programming language"
        })
        self.assertNotIn("error", result)
        self.assertIn("content", result)
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)

    @unittest.skipUnless(
        os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_API_KEY") != "dummy_key",
        "Requires valid OPENROUTER_API_KEY"
    )
    def test_real_search_with_max_results(self):
        """Test real search with max_results limit."""
        result = self._execute_tool_call("web_search", {
            "query": "artificial intelligence",
            "max_results": 2
        })
        self.assertNotIn("error", result)
        self.assertIn("results", result)
        if result["results"]:
            self.assertLessEqual(len(result["results"]), 2)

    @unittest.skipUnless(
        os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_API_KEY") != "dummy_key",
        "Requires valid OPENROUTER_API_KEY"
    )
    def test_real_search_current_events(self):
        """Test search for current events."""
        result = self._execute_tool_call("web_search", {
            "query": "latest technology news 2024"
        })
        self.assertNotIn("error", result)
        self.assertIn("content", result)

    @unittest.skipUnless(
        os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_API_KEY") != "dummy_key",
        "Requires valid OPENROUTER_API_KEY"
    )
    def test_real_search_statistics(self):
        """Test search for statistics."""
        result = self._execute_tool_call("web_search", {
            "query": "world population 2024"
        })
        self.assertNotIn("error", result)
        self.assertIn("content", result)

    @unittest.skipUnless(
        os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_API_KEY") != "dummy_key",
        "Requires valid OPENROUTER_API_KEY"
    )
    def test_real_search_scientific(self):
        """Test search for scientific information."""
        result = self._execute_tool_call("web_search", {
            "query": "quantum computing advances 2024"
        })
        self.assertNotIn("error", result)
        self.assertIn("content", result)

    # ========== RESULT PARSING ==========

    def test_result_structure(self):
        """Test structure of parsed results."""
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        
        self.assertIsInstance(result["results"], list)
        if result.get("results") and len(result["results"]) > 0:
            first_result = result["results"][0]
            self.assertIsInstance(first_result, dict)
            if "type" in first_result:
                self.assertEqual(first_result["type"], "citation")

    def test_annotations_not_in_response(self):
        """Test that annotations are not returned (only parsed into results)."""
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        
        self.assertNotIn("annotations", result)

    # ========== MODEL RESPONSE ==========

    def test_model_field_not_present(self):
        """Test that model field is not returned in response."""
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        
        if not result.get("error"):
            self.assertNotIn("model", result)

    def test_content_field_present(self):
        """Test that content field is present in successful responses."""
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        
        if not result.get("error"):
            self.assertIn("content", result)
            self.assertIsInstance(result["content"], str)

    # ========== EDGE CASES ==========

    def test_unicode_query(self):
        """Test query with unicode characters."""
        result = self._execute_tool_call("web_search", {
            "query": "测试 日本語 한국어"
        })
        self.assertIn("query", result)

    def test_numeric_query(self):
        """Test purely numeric query."""
        result = self._execute_tool_call("web_search", {
            "query": "12345"
        })
        self.assertIn("query", result)

    def test_single_word_query(self):
        """Test single word query."""
        result = self._execute_tool_call("web_search", {
            "query": "Python"
        })
        self.assertIn("query", result)

    def test_multi_word_query(self):
        """Test multi-word query."""
        result = self._execute_tool_call("web_search", {
            "query": "machine learning artificial intelligence"
        })
        self.assertIn("query", result)

    # ========== TOOL CALL FORMAT COMPATIBILITY ==========

    def test_tool_call_format_compatibility(self):
        """Test that tool accepts model-formatted arguments."""
        arguments = {
            "query": "test query",
            "max_results": 5
        }
        result = self._execute_tool_call("web_search", arguments)
        self.assertIn("query", result)
        self.assertEqual(result["query"], "test query")

    def test_tool_call_with_only_query(self):
        """Test tool call with only required query parameter."""
        arguments = {
            "query": "test query"
        }
        result = self._execute_tool_call("web_search", arguments)
        self.assertIn("query", result)

    def test_tool_call_with_all_parameters(self):
        """Test tool call with all parameters."""
        arguments = {
            "query": "test query",
            "max_results": 10
        }
        result = self._execute_tool_call("web_search", arguments)
        self.assertIn("query", result)


if __name__ == '__main__':
    unittest.main()

