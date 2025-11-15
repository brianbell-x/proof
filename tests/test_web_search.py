import unittest
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tool.proof_tool import WebSearchTool, get_search_schema as get_tool_schema


class MockToolCall:
    def __init__(self, tool_name: str, arguments: dict, call_id: str = "test_call_123"):
        self.id = call_id
        self.function = MockFunction(tool_name, arguments)

class MockFunction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)

class TestWebSearchTool(unittest.TestCase):
    def setUp(self):
        api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key")
        self.tool = WebSearchTool(api_key)
        self.has_valid_key = api_key != "dummy_key"

    def _execute_tool_call(self, tool_name: str, arguments: dict) -> dict:
        mock_call = MockToolCall(tool_name, arguments)
        return self.tool.search(**arguments)

    # ========== BASIC FUNCTIONALITY ==========

    def test_initialization(self):
        self.assertIsNotNone(self.tool.api_key)
        self.assertEqual(self.tool.base_url, "https://openrouter.ai/api/v1")

    def test_get_tool_schema(self):
        schema = get_tool_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "web_search")
        self.assertIn("query", schema["function"]["parameters"]["properties"])
        self.assertIn("max_results", schema["function"]["parameters"]["properties"])

    def test_schema_query_required(self):
        schema = get_tool_schema()
        required = schema["function"]["parameters"].get("required", [])
        self.assertIn("query", required)

    def test_schema_max_results_optional(self):
        schema = get_tool_schema()
        required = schema["function"]["parameters"].get("required", [])
        self.assertNotIn("max_results", required)

    # ========== ERROR HANDLING ==========

    def test_invalid_api_key_handling(self):
        invalid_tool = WebSearchTool("invalid_key")
        result = invalid_tool.search("test query")
        self.assertIn("error", result)
        self.assertEqual(result["query"], "test query")
        self.assertIn("timestamp", result)

    def test_empty_query(self):
        result = self._execute_tool_call("web_search", {
            "query": ""
        })
        self.assertIn("query", result)
        self.assertIn("timestamp", result)

    def test_missing_query_parameter(self):
        result = self.tool.search()
        self.assertIn("error", result)

    # ========== RESPONSE STRUCTURE ==========

    def test_response_structure(self):
        result = self._execute_tool_call("web_search", {
            "query": "test query"
        })
        
        required_fields = ["query", "timestamp", "results"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing field: {field}")

    def test_timestamp_format(self):
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        self.assertIn("timestamp", result)
        timestamp = result["timestamp"]
        self.assertIsInstance(timestamp, str)
        self.assertIn("T", timestamp or "2024-01-01T00:00:00")

    def test_results_is_list(self):
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        self.assertIsInstance(result["results"], list)

    # ========== QUERY VARIATIONS ==========

    def test_simple_query(self):
        result = self._execute_tool_call("web_search", {
            "query": "Python programming"
        })
        self.assertIn("query", result)
        self.assertEqual(result["query"], "Python programming")

    def test_query_with_special_characters(self):
        result = self._execute_tool_call("web_search", {
            "query": "C++ programming & development"
        })
        self.assertIn("query", result)
        self.assertEqual(result["query"], "C++ programming & development")

    def test_query_with_numbers(self):
        result = self._execute_tool_call("web_search", {
            "query": "Python 3.12 features"
        })
        self.assertIn("query", result)
        self.assertEqual(result["query"], "Python 3.12 features")

    def test_long_query(self):
        long_query = " ".join(["test"] * 50)
        result = self._execute_tool_call("web_search", {
            "query": long_query
        })
        self.assertIn("query", result)

    def test_query_with_quotes(self):
        result = self._execute_tool_call("web_search", {
            "query": 'What is "machine learning"?'
        })
        self.assertIn("query", result)

    # ========== MAX_RESULTS PARAMETER ==========

    def test_max_results_parameter(self):
        result = self._execute_tool_call("web_search", {
            "query": "test",
            "max_results": 3
        })
        self.assertIn("results", result)
        if not result.get("error"):
            self.assertLessEqual(len(result["results"]), 3)

    def test_max_results_default(self):
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        self.assertIn("results", result)

    def test_max_results_one(self):
        result = self._execute_tool_call("web_search", {
            "query": "test",
            "max_results": 1
        })
        self.assertIn("results", result)

    def test_max_results_large(self):
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
        result = self._execute_tool_call("web_search", {
            "query": "quantum computing advances 2024"
        })
        self.assertNotIn("error", result)
        self.assertIn("content", result)

    # ========== RESULT PARSING ==========

    def test_result_structure(self):
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
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        
        self.assertNotIn("annotations", result)

    # ========== MODEL RESPONSE ==========

    def test_model_field_not_present(self):
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        
        if not result.get("error"):
            self.assertNotIn("model", result)

    def test_content_field_present(self):
        result = self._execute_tool_call("web_search", {
            "query": "test"
        })
        
        if not result.get("error"):
            self.assertIn("content", result)
            self.assertIsInstance(result["content"], str)

    # ========== EDGE CASES ==========

    def test_unicode_query(self):
        result = self._execute_tool_call("web_search", {
            "query": "测试 日本語 한국어"
        })
        self.assertIn("query", result)

    def test_numeric_query(self):
        result = self._execute_tool_call("web_search", {
            "query": "12345"
        })
        self.assertIn("query", result)

    def test_single_word_query(self):
        result = self._execute_tool_call("web_search", {
            "query": "Python"
        })
        self.assertIn("query", result)

    def test_multi_word_query(self):
        result = self._execute_tool_call("web_search", {
            "query": "machine learning artificial intelligence"
        })
        self.assertIn("query", result)

    # ========== TOOL CALL FORMAT COMPATIBILITY ==========

    def test_tool_call_format_compatibility(self):
        arguments = {
            "query": "test query",
            "max_results": 5
        }
        result = self._execute_tool_call("web_search", arguments)
        self.assertIn("query", result)
        self.assertEqual(result["query"], "test query")

    def test_tool_call_with_only_query(self):
        arguments = {
            "query": "test query"
        }
        result = self._execute_tool_call("web_search", arguments)
        self.assertIn("query", result)

    def test_tool_call_with_all_parameters(self):
        arguments = {
            "query": "test query",
            "max_results": 10
        }
        result = self._execute_tool_call("web_search", arguments)
        self.assertIn("query", result)


if __name__ == '__main__':
    unittest.main()

