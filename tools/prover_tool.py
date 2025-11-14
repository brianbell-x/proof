"""
Prover tool for the Prover system.
Contains ProverAgent class and ProverTool wrapper.

This module contains all the logic for claim verification, including:
- ProverAgent: The core agent that constructs rigorous proofs using tools
- ProverTool: A wrapper that makes ProverAgent callable as a tool by other agents
"""

import os
import json
import logging
import time
from typing import Dict, Any
from datetime import datetime
from openai import OpenAI

# Import tools that ProverAgent uses (web_search and python_execute)
from tools.search import WebSearchTool, get_tool_schema as get_search_schema
from tools.python_repl import PythonREPLTool, get_tool_schema as get_python_schema

# Set up logging for ProverAgent
log_dir = 'logs'
log_file = os.path.join(log_dir, 'prover.log')

try:
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w')
except (OSError, PermissionError) as e:
    print(f"Warning: Could not create log file {log_file}: {e}. Logging to console only.")
    file_handler = None

handlers = [logging.StreamHandler()]
if file_handler:
    handlers.append(file_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)

# Configure console verbosity based on PROVER_VERBOSE env var
console_level = logging.WARNING
if os.getenv("PROVER_VERBOSE", "0").lower() in ("1", "true", "yes"):
    console_level = logging.INFO
logging.getLogger().handlers[0].setLevel(console_level)

# Console log truncation for readability (file logs remain full)
PROVER_LOG_MAX_CHARS = int(os.getenv("PROVER_LOG_MAX_CHARS", "0"))  # 0 = no truncation

logger = logging.getLogger(__name__)


def _strip_markdown_code_fences(content: str) -> str:
    """
    Strip markdown code fences (```json or ```) from content.
    
    Args:
        content: Content that may contain markdown code fences
        
    Returns:
        Content with code fences removed
    """
    if not content:
        return content
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    return content.strip()


class ProverAgent:
    """
    Agent that constructs rigorous proofs using tools for evidence gathering.
    
    This agent uses web_search and python_execute tools to verify claims through
    logical analysis and empirical evidence. It follows a falsification-first
    approach, attempting to break claims before defending them.
    """

    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        """
        Initialize the ProverAgent.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use for claim verification (default: grok-4-fast)
        """
        self.api_key = api_key
        self.model = model
        # Initialize OpenAI client for API calls
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Initialize tools that ProverAgent can use
        self.tools = {}
        self.tools["web_search"] = WebSearchTool(api_key)
        self.tools["python_execute"] = PythonREPLTool()

        # Load prompt for the agent (prover_prompt.md contains all prompt content)
        self.master_prompt = self._load_prompt("prompts/prover_prompt.md")
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = f"Current date: {current_date}\n\n" + self.master_prompt

        # Set up tool schemas for the model
        self.tool_schemas = []
        if ":online" not in self.model:
            self.tool_schemas.append(get_search_schema())
        self.tool_schemas.append(get_python_schema())

    def _load_prompt(self, path: str) -> str:
        """
        Load a prompt file.
        
        Args:
            path: Path to the prompt file
            
        Returns:
            Contents of the prompt file, or empty string if not found
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {path}")
            return ""

    def _calculate_costs(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        """
        Calculate API costs based on token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Dict with cost breakdown in USD
        """
        input_cost_per_million = 0.20
        output_cost_per_million = 0.50
        input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
        output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
        total_cost = input_cost + output_cost
        return {
            "input_usd": round(input_cost, 6),
            "output_usd": round(output_cost, 6),
            "total_usd": round(total_cost, 6)
        }

    def _execute_tool(self, tool_call) -> Dict[str, Any]:
        """
        Execute a tool call requested by the model.
        
        Args:
            tool_call: Tool call object from the API response
            
        Returns:
            Dict containing tool execution results
        """
        start_time = time.time()

        try:
            # Extract tool call information from different possible formats
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id
            elif isinstance(tool_call, dict):
                tool_name = tool_call.get('function', {}).get('name')
                arguments = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                tool_call_id = tool_call.get('id')
            else:
                raise ValueError(f"Unexpected tool_call format: {type(tool_call)}")

            logger.info(f"[TOOL CALL REQUEST][id={tool_call_id}][name={tool_name}] args={json.dumps(arguments)}")

            # Execute the appropriate tool
            if tool_name not in self.tools:
                result = {
                    "error": f"Unknown tool: {tool_name}",
                    "tool_call_id": tool_call_id
                }
            else:
                tool = self.tools[tool_name]

                if tool_name == "web_search" and tool_name in self.tools:
                    result = tool.search(**arguments)
                elif tool_name == "python_execute":
                    result = tool.execute(**arguments)
                else:
                    result = {"error": f"Tool {tool_name} not implemented or not available for this model"}

                result["tool_call_id"] = tool_call_id
                result["tool_name"] = tool_name

            # Log tool execution results
            duration = time.time() - start_time
            result_payload = json.dumps(result)
            if PROVER_LOG_MAX_CHARS > 0 and len(result_payload) > PROVER_LOG_MAX_CHARS:
                truncated_payload = result_payload[:PROVER_LOG_MAX_CHARS] + "... (truncated)"
                logger.info(f"[TOOL RESULT][id={tool_call_id}][name={tool_name}][duration={duration:.3f}s] result={truncated_payload}")
            else:
                logger.info(f"[TOOL RESULT][id={tool_call_id}][name={tool_name}][duration={duration:.3f}s] result={result_payload}")

            return result

        except Exception as e:
            # Handle errors during tool execution
            duration = time.time() - start_time
            tool_call_id = getattr(tool_call, 'id', None) if hasattr(tool_call, 'id') else (tool_call.get('id') if isinstance(tool_call, dict) else None)
            tool_name = None
            if hasattr(tool_call, 'function'):
                tool_name = getattr(tool_call.function, 'name', None)
            elif isinstance(tool_call, dict):
                tool_name = tool_call.get('function', {}).get('name')
            logger.error(f"[TOOL RESULT][id={tool_call_id or 'unknown'}][name={tool_name or 'unknown'}][duration={duration:.3f}s] error={str(e)}")
            return {
                "error": str(e),
                "tool_call_id": tool_call_id,
                "tool_name": tool_name
            }

    def prove_claim(self, claim: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Verify a claim through rigorous logical analysis and evidence gathering.
        
        This is the main method that orchestrates the claim verification process.
        It uses an iterative approach, making API calls and tool calls until a
        verdict is reached.
        
        Args:
            claim: The claim to verify
            max_iterations: Maximum number of iterations (optional, defaults to unlimited)
            
        Returns:
            Dict containing the verification result with verdict, reasoning, evidence, etc.
        """
        start_time = time.time()
        # Initialize conversation with system prompt and user claim
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": claim
            }
        ]

        iteration = 0
        final_result = None
        tools_used = set()
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Main verification loop - continues until verdict is reached
        while True:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")

            try:
                # Make API call to the model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto"
                )

                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    total_prompt_tokens += getattr(usage, 'prompt_tokens', 0)
                    total_completion_tokens += getattr(usage, 'completion_tokens', 0)

                # Extract message from response
                message = response.choices[0].message
                message_dict = message.model_dump() if hasattr(message, 'model_dump') else message
                messages.append(message_dict)

                # Check for tool calls
                tool_calls = getattr(message, 'tool_calls', None)
                if not tool_calls and isinstance(message_dict, dict):
                    tool_calls = message_dict.get('tool_calls')

                # Log model output
                finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                model_used = getattr(response, 'model', self.model)
                content = message.content or ""
                tool_calls_info = ""
                if tool_calls:
                    tool_calls_list = []
                    for tc in tool_calls:
                        if hasattr(tc, 'function'):
                            tool_calls_list.append(f"id={tc.id} name={tc.function.name} args={tc.function.arguments}")
                        elif isinstance(tc, dict):
                            tool_calls_list.append(f"id={tc.get('id', 'unknown')} name={tc.get('function', {}).get('name', 'unknown')}")
                    tool_calls_info = f" tool_calls=[{', '.join(tool_calls_list)}]"

                if "verdict" in content:
                    logger.info(f"[MODEL OUTPUT][iter={iteration}][finish_reason={finish_reason}][model={model_used}] content={content}{tool_calls_info}")
                else:
                    logger.info(f"[MODEL OUTPUT][iter={iteration}][finish_reason={finish_reason}][model={model_used}] content={content}{tool_calls_info}")

                # Try to parse JSON response
                try:
                    cleaned_content = _strip_markdown_code_fences(content)
                    result = json.loads(cleaned_content) if cleaned_content else {}
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"[MODEL OUTPUT][parse_error] Failed to parse JSON response: {content}")
                    result = {"error": "Invalid JSON response", "raw_content": content}

                # Check if verdict was reached
                if "verdict" in result:
                    final_result = result
                    logger.info("[FINAL RESULT] Verdict reached")
                    break

                # Process tool calls if any
                if tool_calls:
                    logger.info(f"Processing {len(tool_calls)} tool calls")

                    for tool_call in tool_calls:
                        # Extract tool call information
                        if hasattr(tool_call, 'function'):
                            tool_name = tool_call.function.name
                            tool_call_id = tool_call.id
                        elif isinstance(tool_call, dict):
                            tool_name = tool_call.get('function', {}).get('name')
                            tool_call_id = tool_call.get('id')
                        else:
                            logger.warning(f"Unexpected tool_call format: {type(tool_call)}")
                            continue

                        tools_used.add(tool_name)
                        # Execute the tool
                        tool_result = self._execute_tool(tool_call)

                        # Add tool result to conversation
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(tool_result)
                        }
                        messages.append(tool_message)
                        logger.info(f"[TOOL RESULT SENT][id={tool_call_id}] message={json.dumps(tool_message)}")

                else:
                    logger.info("No tool calls in response, continuing...")

            except Exception as e:
                # Handle errors during iteration
                logger.error(f"Error in iteration {iteration}: {e}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                cost_info = self._calculate_costs(total_prompt_tokens, total_completion_tokens)
                return {
                    "error": str(e),
                    "claim": claim,
                    "iterations_completed": iteration,
                    "time_seconds": round(elapsed_time, 3),
                    "tokens": {
                        "prompt": total_prompt_tokens,
                        "completion": total_completion_tokens,
                        "total": total_prompt_tokens + total_completion_tokens
                    },
                    "cost": cost_info
                }

        # Calculate final statistics
        end_time = time.time()
        elapsed_time = end_time - start_time
        cost_info = self._calculate_costs(total_prompt_tokens, total_completion_tokens)
        tokens_info = {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_prompt_tokens + total_completion_tokens
        }

        # Return final result with metadata
        if final_result:
            final_result["iterations_used"] = iteration
            final_result["tools_used"] = list(tools_used)
            final_result["time_seconds"] = round(elapsed_time, 3)
            final_result["tokens"] = tokens_info
            final_result["cost"] = cost_info
            logger.info(f"[FINAL RESULT] {json.dumps(final_result)}")
            return final_result
        else:
            # Return partial result if no verdict reached
            partial = {
                "error": "No verdict reached",
                "claim": claim,
                "iterations_completed": iteration,
                "partial_result": result if 'result' in locals() else None,
                "tools_used": list(tools_used),
                "time_seconds": round(elapsed_time, 3),
                "tokens": tokens_info,
                "cost": cost_info
            }
            logger.info(f"[FINAL RESULT] {json.dumps(partial)}")
            return partial


class ProverTool:
    """
    A tool that wraps ProverAgent functionality.
    Allows other agents to use claim verification capabilities as a tool.
    
    This class provides a simple interface for other agents (like ChatAgent)
    to use ProverAgent's claim verification functionality without needing
    to directly instantiate ProverAgent.
    """

    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        """
        Initialize the Prover tool.
        
        Args:
            api_key: OpenRouter API key for the ProverAgent
            model: Model to use for the ProverAgent (default: grok-4-fast)
        """
        # Create an instance of ProverAgent to handle claim verification
        self.prover_agent = ProverAgent(api_key, model)

    def prove_claim(self, claim: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Verify a claim using the ProverAgent.
        
        This method wraps the ProverAgent's prove_claim method to make it
        callable as a tool by other agents.
        
        Args:
            claim: The claim to verify
            max_iterations: Maximum number of iterations for the prover (optional)
            
        Returns:
            Dict containing the verification result with verdict, reasoning, evidence, etc.
        """
        # Call the ProverAgent's prove_claim method
        result = self.prover_agent.prove_claim(claim, max_iterations)
        
        # Add timestamp to the result
        result["timestamp"] = datetime.now().isoformat()
        
        return result


def get_tool_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for the Prover tool.
    
    This schema defines how the tool appears to the LLM when making tool calls.
    
    Returns:
        Tool schema dictionary compatible with OpenAI function calling format
    """
    return {
        "type": "function",
        "function": {
            "name": "prove_claim",
            "description": "Verify whether a claim is true or false by stress-testing it through rigorous logical analysis and evidence gathering. Use this tool when the user wants to know if a claim is true or false, or when they ask about verifying a statement. The tool will return a verdict (PROVEN, DISPROVEN, UNSUPPORTED, or UNVERIFIABLE) along with detailed reasoning, evidence, and derivation steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The claim or statement to verify. This should be a clear, specific statement that can be evaluated."
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum number of reasoning iterations (optional, defaults to unlimited)",
                        "minimum": 1
                    }
                },
                "required": ["claim"]
            }
        }
    }
