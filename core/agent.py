"""
This module implements an agentic reasoning system that can use tools to gather evidence
and perform calculations while constructing rigorous proofs for claims.
"""

import os
import json
import logging
import time
from typing import Dict, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from tools.search import WebSearchTool, get_tool_schema as get_search_schema
from tools.python_repl import PythonREPLTool, get_tool_schema as get_python_schema

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file or environment variables.")

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
    """Strip markdown code fences (```json or ```) from content."""
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
    """Agent that constructs rigorous proofs using tools for evidence gathering."""

    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.tools = {}
        self.tools["web_search"] = WebSearchTool(api_key)
        self.tools["python_execute"] = PythonREPLTool()

        self.master_prompt = self._load_prompt("prompts/master.md")
        self.tool_prompt = self._load_prompt("prompts/tool_prompt.md")
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = f"Current date: {current_date}\n\n" + self.master_prompt + "\n\n" + self.tool_prompt

        self.tool_schemas = []
        if ":online" not in self.model:
            self.tool_schemas.append(get_search_schema())
        self.tool_schemas.append(get_python_schema())

    def _load_prompt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {path}")
            return ""

    def _calculate_costs(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
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
        import time
        start_time = time.time()

        try:
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

            duration = time.time() - start_time
            result_payload = json.dumps(result)
            if PROVER_LOG_MAX_CHARS > 0 and len(result_payload) > PROVER_LOG_MAX_CHARS:
                truncated_payload = result_payload[:PROVER_LOG_MAX_CHARS] + "... (truncated)"
                logger.info(f"[TOOL RESULT][id={tool_call_id}][name={tool_name}][duration={duration:.3f}s] result={truncated_payload}")
            else:
                logger.info(f"[TOOL RESULT][id={tool_call_id}][name={tool_name}][duration={duration:.3f}s] result={result_payload}")

            return result

        except Exception as e:
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
        start_time = time.time()
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

        while True:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto"
                )

                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    total_prompt_tokens += getattr(usage, 'prompt_tokens', 0)
                    total_completion_tokens += getattr(usage, 'completion_tokens', 0)

                message = response.choices[0].message
                message_dict = message.model_dump() if hasattr(message, 'model_dump') else message
                messages.append(message_dict)

                tool_calls = getattr(message, 'tool_calls', None)
                if not tool_calls and isinstance(message_dict, dict):
                    tool_calls = message_dict.get('tool_calls')

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

                try:
                    cleaned_content = _strip_markdown_code_fences(content)
                    result = json.loads(cleaned_content) if cleaned_content else {}
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"[MODEL OUTPUT][parse_error] Failed to parse JSON response: {content}")
                    result = {"error": "Invalid JSON response", "raw_content": content}

                if "verdict" in result:
                    final_result = result
                    logger.info("[FINAL RESULT] Verdict reached")
                    break

                if tool_calls:
                    logger.info(f"Processing {len(tool_calls)} tool calls")

                    for tool_call in tool_calls:
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
                        tool_result = self._execute_tool(tool_call)

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

        end_time = time.time()
        elapsed_time = end_time - start_time
        cost_info = self._calculate_costs(total_prompt_tokens, total_completion_tokens)
        tokens_info = {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_prompt_tokens + total_completion_tokens
        }

        if final_result:
            final_result["iterations_used"] = iteration
            final_result["tools_used"] = list(tools_used)
            final_result["time_seconds"] = round(elapsed_time, 3)
            final_result["tokens"] = tokens_info
            final_result["cost"] = cost_info
            logger.info(f"[FINAL RESULT] {json.dumps(final_result)}")
            return final_result
        else:
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


def main():
    print("Proof Agent")
    print("====================")
    print("Enter a claim to analyze (or 'quit' to exit):")
    print("Set PROVER_VERBOSE=1 to see detailed logs in console.")
    print("Set PROVER_SHOW_FULL=1 to print full JSON results to console.")

    agent = ProverAgent(api_key)

    while True:
        try:
            user_input = input("\nClaim: ").strip()

            if user_input.lower() in ['quit']:
                logger.info("User requested exit")
                break

            if not user_input:
                continue

            logger.info(f"Analyzing claim: {user_input}")

            result = agent.prove_claim(user_input)

            logger.info("Analysis complete")

            # Pretty CLI summary
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
                if "partial_result" in result:
                    print("Partial result available in logs.")
                if "time_seconds" in result:
                    print(f"   Time: {result['time_seconds']}s")
                if "cost" in result:
                    print(f"   Cost: ${result['cost']['total_usd']:.6f}")
            else:
                verdict = result.get("verdict", "UNKNOWN")
                reason = result.get("reasoning_summary", "No summary available")
                iterations = result.get("iterations_used", "?")
                tools = result.get("tools_used", [])
                time_seconds = result.get("time_seconds", 0)
                cost_info = result.get("cost", {})
                tokens_info = result.get("tokens", {})

                verdict_emoji = {"PROVEN": "✅", "DISPROVEN": "❌", "UNSUPPORTED": "❓", "UNVERIFIABLE": "🤷"}.get(verdict, "❓")

                print(f"\n{verdict_emoji} Verdict: {verdict}")
                print(f"   Reason: {reason}")
                print(f"   Iterations: {iterations}")
                print(f"   Tools used: {', '.join(tools) if tools else 'None'}")
                print(f"   Time: {time_seconds}s")
                if cost_info:
                    print(f"   Cost: ${cost_info.get('total_usd', 0):.6f} (input: ${cost_info.get('input_usd', 0):.6f}, output: ${cost_info.get('output_usd', 0):.6f})")
                if tokens_info:
                    print(f"   Tokens: {tokens_info.get('total', 0)} (prompt: {tokens_info.get('prompt', 0)}, completion: {tokens_info.get('completion', 0)})")

            # Optional full JSON output
            if os.getenv("PROVER_SHOW_FULL", "0").lower() in ("1", "true", "yes"):
                print("\nFull result:")
                print(json.dumps(result, indent=2))

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, exiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred in main loop: {e}")


if __name__ == "__main__":
    main()