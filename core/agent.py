"""
This module implements an agentic reasoning system that can use tools to gather evidence
and perform calculations while constructing rigorous proofs for claims.
"""

import os
import json
import logging
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
    file_handler = logging.FileHandler(log_file)
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

    def _execute_tool(self, tool_call) -> Dict[str, Any]:
        import time
        start_time = time.time()

        try:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            logger.info(f"[TOOL CALL REQUEST][id={tool_call.id}][name={tool_name}] args={json.dumps(arguments)}")

            if tool_name not in self.tools:
                result = {
                    "error": f"Unknown tool: {tool_name}",
                    "tool_call_id": tool_call.id
                }
            else:
                tool = self.tools[tool_name]

                if tool_name == "web_search" and tool_name in self.tools:
                    result = tool.search(**arguments)
                elif tool_name == "python_execute":
                    result = tool.execute(**arguments)
                else:
                    result = {"error": f"Tool {tool_name} not implemented or not available for this model"}

                result["tool_call_id"] = tool_call.id
                result["tool_name"] = tool_name

            duration = time.time() - start_time
            result_payload = json.dumps(result)
            if PROVER_LOG_MAX_CHARS > 0 and len(result_payload) > PROVER_LOG_MAX_CHARS:
                truncated_payload = result_payload[:PROVER_LOG_MAX_CHARS] + "... (truncated)"
                logger.info(f"[TOOL RESULT][id={tool_call.id}][name={tool_name}][duration={duration:.3f}s] result={truncated_payload}")
            else:
                logger.info(f"[TOOL RESULT][id={tool_call.id}][name={tool_name}][duration={duration:.3f}s] result={result_payload}")

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[TOOL RESULT][id={getattr(tool_call, 'id', 'unknown')}][name={getattr(tool_call.function, 'name', 'unknown') if hasattr(tool_call, 'function') else 'unknown'}][duration={duration:.3f}s] error={str(e)}")
            return {
                "error": str(e),
                "tool_call_id": getattr(tool_call, "id", None),
                "tool_name": getattr(tool_call.function, "name", None) if hasattr(tool_call, "function") else None
            }

    def prove_claim(self, claim: str, max_iterations: int = 5) -> Dict[str, Any]:
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

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto"
                )

                message = response.choices[0].message
                message_dict = message.model_dump() if hasattr(message, 'model_dump') else message
                messages.append(message_dict)

                # Log model output with full details
                finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                model_used = getattr(response, 'model', self.model)
                content = message.content or ""
                tool_calls_info = ""
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls_list = []
                    for tc in message.tool_calls:
                        tool_calls_list.append(f"id={tc.id} name={tc.function.name} args={tc.function.arguments}")
                    tool_calls_info = f" tool_calls=[{', '.join(tool_calls_list)}]"

                if "verdict" in content:
                    logger.info(f"[MODEL OUTPUT][iter={iteration}][finish_reason={finish_reason}][model={model_used}] content={content}{tool_calls_info}")
                else:
                    logger.info(f"[MODEL OUTPUT][iter={iteration}][finish_reason={finish_reason}][model={model_used}] content={content}{tool_calls_info}")

                try:
                    result = json.loads(content) if content else {}
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"[MODEL OUTPUT][parse_error] Failed to parse JSON response: {content}")
                    result = {"error": "Invalid JSON response", "raw_content": content}

                if "verdict" in result:
                    final_result = result
                    logger.info("[FINAL RESULT] Verdict reached")
                    break

                if hasattr(message, 'tool_calls') and message.tool_calls:
                    logger.info(f"Processing {len(message.tool_calls)} tool calls")

                    for tool_call in message.tool_calls:
                        tools_used.add(tool_call.function.name)
                        tool_result = self._execute_tool(tool_call)

                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result)
                        }
                        messages.append(tool_message)
                        logger.info(f"[TOOL RESULT SENT][id={tool_call.id}] message={json.dumps(tool_message)}")

                else:
                    logger.info("No tool calls in response, continuing...")

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                return {
                    "error": str(e),
                    "claim": claim,
                    "iterations_completed": iteration
                }

        if final_result:
            final_result["iterations_used"] = iteration
            final_result["tools_used"] = list(tools_used)
            logger.info(f"[FINAL RESULT] {json.dumps(final_result)}")
            return final_result
        else:
            partial = {
                "error": "Maximum iterations reached without reaching a verdict",
                "claim": claim,
                "iterations_completed": iteration,
                "partial_result": result if 'result' in locals() else None,
                "tools_used": list(tools_used)
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
            else:
                verdict = result.get("verdict", "UNKNOWN")
                confidence = result.get("confidence", "UNKNOWN")
                reason = result.get("reasoning_summary", "No summary available")
                iterations = result.get("iterations_used", "?")
                tools = result.get("tools_used", [])

                verdict_emoji = {"PROVEN": "✅", "DISPROVEN": "❌", "UNSUPPORTED": "❓", "UNVERIFIABLE": "🤷"}.get(verdict, "❓")

                print(f"\n{verdict_emoji} Verdict: {verdict}")
                print(f"   Confidence: {confidence}")
                print(f"   Reason: {reason}")
                print(f"   Iterations: {iterations}")
                print(f"   Tools used: {', '.join(tools) if tools else 'None'}")

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