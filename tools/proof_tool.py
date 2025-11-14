import os
import json
import logging
import time
from typing import Dict, Any
from datetime import datetime
from openai import OpenAI
from tools.search import WebSearchTool, get_tool_schema as get_search_schema
from tools.code_execution import CodeExecutionTool, get_tool_schema as get_python_schema

log_dir = 'logs'
log_file = os.path.join(log_dir, 'proof.log')

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

console_level = logging.WARNING
if os.getenv("PROOF_VERBOSE", "0").lower() in ("1", "true", "yes"):
    console_level = logging.INFO
logging.getLogger().handlers[0].setLevel(console_level)

PROOF_LOG_MAX_CHARS = int(os.getenv("PROOF_LOG_MAX_CHARS", "0"))
logger = logging.getLogger(__name__)


def _strip_markdown_code_fences(content: str) -> str:
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


class ProofAgent:
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.tools = {
            "web_search": WebSearchTool(api_key),
            "python_execute": CodeExecutionTool()
        }
        self.master_prompt = self._load_prompt("prompts/proof_prompt.md")
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        # Only substitute the known placeholders to avoid interpreting JSON braces as format fields
        self.system_prompt = self._interpolate(self.master_prompt, {
            "current_date": current_date,
            "current_time": current_time
        })
        self.tool_schemas = [get_search_schema(), get_python_schema()]

    def _load_prompt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {path}")
            return ""

    def _interpolate(self, text: str, values: Dict[str, str]) -> str:
        """
        Safely interpolate only the placeholders we control (e.g., {current_date}, {current_time})
        without invoking Python's format() on the whole prompt. This preserves literal braces
        in JSON examples contained in the prompt.
        """
        if not text:
            return text
        for k, v in values.items():
            text = text.replace("{" + k + "}", v)
        return text

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
                if tool_name == "web_search":
                    result = tool.search(**arguments)
                elif tool_name == "python_execute":
                    result = tool.execute(**arguments)
                else:
                    result = {"error": f"Tool {tool_name} not implemented"}

                result["tool_call_id"] = tool_call_id
                result["tool_name"] = tool_name

            duration = time.time() - start_time
            result_payload = json.dumps(result)
            if PROOF_LOG_MAX_CHARS > 0 and len(result_payload) > PROOF_LOG_MAX_CHARS:
                truncated_payload = result_payload[:PROOF_LOG_MAX_CHARS] + "... (truncated)"
                logger.info(f"[TOOL RESULT][id={tool_call_id}][name={tool_name}][duration={duration:.3f}s] result={truncated_payload}")
            else:
                logger.info(f"[TOOL RESULT][id={tool_call_id}][name={tool_name}][duration={duration:.3f}s] result={result_payload}")

            return result

        except (ValueError, json.JSONDecodeError, KeyError) as e:
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
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": claim}
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

                # Fallback: some models embed "tool_calls" inside the JSON content instead of using real function calls.
                # If present, execute them as if they were proper tool calls, append the tool messages, and continue the loop.
                if not tool_calls and isinstance(result, dict) and isinstance(result.get("tool_calls"), list):
                    embedded_calls = result.get("tool_calls", [])
                    logger.info(f"Processing {len(embedded_calls)} embedded tool_calls from content")
                    for emb in embedded_calls:
                        if isinstance(emb, dict) and isinstance(emb.get("function"), dict):
                            emb_id = emb.get("id") or f"embedded_{int(time.time() * 1000)}"
                            emb_name = emb.get("function", {}).get("name")
                            emb_args = emb.get("function", {}).get("arguments", "{}")
                            # Construct a dict matching the shape expected by _execute_tool
                            emb_tool_call = {
                                "id": emb_id,
                                "type": "function",
                                "function": {
                                    "name": emb_name,
                                    "arguments": emb_args
                                }
                            }
                            try:
                                if emb_name:
                                    tools_used.add(emb_name)
                                tool_result = self._execute_tool(emb_tool_call)
                                tool_message = {
                                    "role": "tool",
                                    "tool_call_id": emb_id,
                                    "content": json.dumps(tool_result)
                                }
                                messages.append(tool_message)
                                logger.info(f"[EMBEDDED TOOL RESULT SENT][id={emb_id}] message={json.dumps(tool_message)}")
                            except Exception as e:
                                logger.error(f"[EMBEDDED TOOL ERROR][id={emb_id} name={emb_name}] {e}")
                        else:
                            logger.warning(f"Skipping invalid embedded tool call structure: {type(emb)}")
                    # After injecting tool results, continue to let the model consume them
                    continue

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


class ProofTool:
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.proof_agent = ProofAgent(api_key, model)

    def prove_claim(self, claim: str, max_iterations: int = None) -> Dict[str, Any]:
        result = self.proof_agent.prove_claim(claim, max_iterations)
        result["timestamp"] = datetime.now().isoformat()
        return result


def get_tool_schema() -> Dict[str, Any]:
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

# ---------------------------------------
# Inline harness to verify tool-call behavior
# ---------------------------------------
if __name__ == "__main__":
    """
    Minimal inline test harness to check whether the ProofAgent actually
    emits python_execute tool calls with executable code (the "model-style" behavior).

    Usage examples:
      - python -m tools.proof_tool "2025 is prime" --max-iter 3
      - python -m tools.proof_tool "Binary search is O(n)" --max-iter 3
      - python -m tools.proof_tool "Softmax isn't a probability distribution" --max-iter 3

    Notes:
      - Requires OPENROUTER_API_KEY in environment. If absent, this will exit gracefully.
      - Prints captured tool call details (including a preview of the code string) and the final JSON result.
    """
    import os
    import sys
    import json
    import argparse
    import logging

    # Load .env if available (optional)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[INLINE TEST] OPENROUTER_API_KEY not set. Skipping live test.")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Inline ProofAgent tool-call verifier")
    parser.add_argument("claim", nargs="*", help="Claim text to analyze")
    parser.add_argument("--max-iter", type=int, default=4, help="Maximum reasoning iterations")
    args = parser.parse_args()

    claim_text = " ".join(args.claim).strip() or "Binary search is O(n)."

    # Create agent
    agent = ProofAgent(api_key)

    # Ensure console logs show tool call details
    try:
        for h in logging.getLogger().handlers:
            h.setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)
    except Exception:
        pass

    # Capture tool calls to inspect code arguments
    tool_events = []
    _original_execute_tool = agent._execute_tool  # bound method

    def _wrapper_execute_tool(tool_call, _orig=_original_execute_tool):
        try:
            if hasattr(tool_call, "function"):
                tool_name = getattr(tool_call.function, "name", "unknown")
                tool_call_id = getattr(tool_call, "id", None)
                arguments_raw = getattr(tool_call.function, "arguments", "{}")
                arguments = json.loads(arguments_raw or "{}")
            elif isinstance(tool_call, dict):
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_call_id = tool_call.get("id")
                arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}") or "{}")
            else:
                tool_name = "unknown"
                tool_call_id = None
                arguments = {}
        except Exception as e:
            print(f"[INLINE TEST] Error parsing tool_call: {e}")
            tool_name = "unknown"
            tool_call_id = None
            arguments = {}

        code_snippet = arguments.get("code", "")
        preview = ""
        try:
            if isinstance(code_snippet, str):
                preview = code_snippet[:400].replace("\n", "\\n")
        except Exception:
            preview = "<unavailable>"

        print("\n[INLINE TEST] TOOL CALL")
        print(f"  id={tool_call_id} name={tool_name}")
        if tool_name == "python_execute":
            print(f"  code_length={len(code_snippet) if isinstance(code_snippet, str) else 0}")
            print(f"  code_preview='{preview}'")

        result = _orig(tool_call)
        try:
            keys = list(result.keys()) if isinstance(result, dict) else []
        except Exception:
            keys = []
        print("[INLINE TEST] TOOL RESULT keys:", keys)
        tool_events.append({
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments_keys": list(arguments.keys()),
            "has_code": bool(arguments.get("code")),
            "result_keys": keys,
        })
        return result

    # Monkey-patch for capture
    agent._execute_tool = _wrapper_execute_tool  # type: ignore[attr-defined]

    print(f"[INLINE TEST] Running claim: {claim_text}")
    final = agent.prove_claim(claim_text, max_iterations=args.max_iter)

    print("\n[INLINE TEST] FINAL OUTPUT")
    try:
        print(json.dumps(final, indent=2))
    except Exception:
        print(final)

    # Quick summary of tool usage
    print("\n[INLINE TEST] TOOL USAGE SUMMARY")
    for evt in tool_events:
        print(f"  id={evt['tool_call_id']} name={evt['tool_name']} has_code={evt['has_code']} result_keys={evt['result_keys']}")
