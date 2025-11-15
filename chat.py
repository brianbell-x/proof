import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tool.proof_tool import ProofTool, get_tool_schema as get_proof_schema

load_dotenv()

logger = logging.getLogger(__name__)

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agent-log.md")
os.makedirs(LOG_DIR, exist_ok=True)

def _write_log(message: str) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message)
            f.flush()
    except (OSError, PermissionError):
        pass


def _format_json_content(content: str) -> str:
    if not content:
        return content
    if content.strip().startswith("```"):
        return content
    try:
        cleaned = _strip_markdown_code_fences(content)
        json.loads(cleaned)
        return f"```json\n{content}\n```"
    except (json.JSONDecodeError, TypeError):
        return content


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

class ChatAgent:
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.proof_tool = ProofTool(api_key, model)
        self.model = model
        self.tool_schemas = [get_proof_schema()]
        
        now = datetime.now()
        self.system_prompt = self._load_prompt("prompts/chat_prompt.md").format(
            current_date=now.strftime("%Y-%m-%d"),
            current_time=now.strftime("%H:%M:%S")
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._new_session = True

    def _load_prompt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {path}")
            _write_log(f"## ERROR \n\n ### type \n Prompt file not found \n\n ### path \n {path}\n\n")
            return "You are a helpful assistant."

    def chat(self, user_message: str) -> Dict[str, Any]:
        if self._new_session:
            file_exists = os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                if file_exists:
                    f.write(f"\n\n# Chat Session\n\n## User Message\n{user_message}\n\n")
                else:
                    f.write(f"# Chat Session\n\n## User Message\n{user_message}\n\n")
            self._new_session = False
        else:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"## User Message\n{user_message}\n\n")
        
        start_time = time.time()
        self.messages.append({"role": "user", "content": user_message})
        
        prompt_tokens = 0
        completion_tokens = 0
        
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                )
                
                usage = getattr(response, "usage", None)
                if usage:
                    prompt_tokens += getattr(usage, "prompt_tokens", 0)
                    completion_tokens += getattr(usage, "completion_tokens", 0)
                
                message = response.choices[0].message
                message_dict = message.model_dump() if hasattr(message, "model_dump") else message
                self.messages.append(message_dict)
                
                tool_calls = getattr(message, "tool_calls", None) or message_dict.get("tool_calls")
                content = message.content or ""
                
                if not tool_calls:
                    result = self._build_result(content, prompt_tokens, completion_tokens, start_time)
                    self._log_model_output(content, tool_calls, result)
                    return result
                
                self._log_model_output(content, tool_calls)
                
                logger.info(f"Processing {len(tool_calls)} tool calls")
                
                for tool_call in tool_calls:
                    tool_result, pt, ct = self._handle_tool_call(tool_call)
                    prompt_tokens += pt
                    completion_tokens += ct
                    
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_result["tool_call_id"],
                        "content": json.dumps(tool_result),
                    }
                    self.messages.append(tool_message)
                
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                _write_log(f"## ERROR \n\n ### error \n {e}\n\n")
                result = self._build_error_result(str(e), prompt_tokens, completion_tokens, start_time)
                self._log_model_output("", None, result)
                return result

    def _handle_tool_call(self, tool_call: Any) -> Tuple[Dict[str, Any], int, int]:
        tool_call_start = time.time()
        
        if hasattr(tool_call, "function"):
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
        elif isinstance(tool_call, dict):
            tool_name = tool_call.get("function", {}).get("name")
            arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            tool_call_id = tool_call.get("id")
        else:
            logger.warning(f"Unexpected tool_call format: {type(tool_call)}")
            _write_log(f"## WARNING \n\n ### Unexpected tool_call format \n {type(tool_call)}\n\n")
            return ({"error": "Invalid tool_call format"}, 0, 0)
        
        _write_log(f"## TOOL CALL REQUEST \n\n ### id \n {tool_call_id} \n\n ### name \n {tool_name} \n\n ### args \n```json\n{json.dumps(arguments, indent=2)}\n```\n\n")
        
        if tool_name == "prove_claim":
            logger.info(f"[TOOL CALL] prove_claim with claim: {arguments.get('claim', '')[:50]}...")
            tool_result = self.proof_tool.prove_claim(
                claim=arguments.get("claim", ""),
                max_iterations=arguments.get("max_iterations"),
            )
            proof_tokens = tool_result.get("tokens", {})
            pt = proof_tokens.get("prompt", 0)
            ct = proof_tokens.get("completion", 0)
            logger.info(f"[COST TRACKING] Added proof tokens: prompt={pt}, completion={ct}")
        else:
            tool_result = {"error": f"Unknown tool: {tool_name}"}
            pt, ct = 0, 0
        
        tool_result["tool_call_id"] = tool_call_id
        tool_result["tool_name"] = tool_name
        duration = time.time() - tool_call_start
        
        _write_log(f"## TOOL RESULT \n\n ### id \n {tool_call_id} \n\n ### name \n {tool_name} \n\n ### duration \n {duration:.3f}s \n\n ### result \n```json\n{json.dumps(tool_result, indent=2)}\n```\n\n")
        
        return (tool_result, pt, ct)

    def _log_model_output(self, content: str, tool_calls: Optional[List[Any]], result: Optional[Dict[str, Any]] = None) -> None:
        tool_calls_info = ""
        if tool_calls:
            tool_calls_list = []
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    tool_calls_list.append(f"id={tc.id} name={tc.function.name} args={tc.function.arguments}")
                elif isinstance(tc, dict):
                    tool_calls_list.append(f"id={tc.get('id', 'unknown')} name={tc.get('function', {}).get('name', 'unknown')}")
            tool_calls_info = f" tool_calls=[{', '.join(tool_calls_list)}]"
        
        formatted_content = _format_json_content(content)
        
        if result:
            result_json = f"```json\n{json.dumps(result, indent=2)}\n```"
            _write_log(f"## MODEL OUTPUT \n\n {result_json}\n\n")
        else:
            _write_log(f"## MODEL OUTPUT \n\n {formatted_content}{tool_calls_info}\n\n")

    def _build_result(self, content: str, prompt_tokens: int, completion_tokens: int, start_time: float) -> Dict[str, Any]:
        cost_info = self._calculate_costs(prompt_tokens, completion_tokens)
        elapsed_time = time.time() - start_time
        result = {
            "response": content,
            "time_seconds": round(elapsed_time, 3),
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            "cost": cost_info,
        }
        return result

    def _build_error_result(self, error: str, prompt_tokens: int, completion_tokens: int, start_time: float) -> Dict[str, Any]:
        cost_info = self._calculate_costs(prompt_tokens, completion_tokens)
        elapsed_time = time.time() - start_time
        result = {
            "error": error,
            "time_seconds": round(elapsed_time, 3),
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            "cost": cost_info,
        }
        return result

    def _calculate_costs(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
        INPUT_COST_PER_MILLION = 0.20
        OUTPUT_COST_PER_MILLION = 0.50
        
        input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_MILLION
        output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
        
        return {
            "input_usd": round(input_cost, 6),
            "output_usd": round(output_cost, 6),
            "total_usd": round(input_cost + output_cost, 6),
        }

def chat_main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file or environment variables.")
    
    chat_agent = ChatAgent(api_key)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            logger.info(f"User message: {user_input}")
            result = chat_agent.chat(user_input)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
                if "time_seconds" in result:
                    print(f"   Time: {result['time_seconds']}s")
                if "cost" in result:
                    print(f"   Cost: ${result['cost']['total_usd']:.6f}")
            else:
                print(f"\nAssistant: {result.get('response', 'No response generated')}")
                print(f"   Time: {result.get('time_seconds', 0)}s")
                cost_info = result.get("cost", {})
                if cost_info:
                    print(f"   Cost: ${cost_info.get('total_usd', 0):.6f}")
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, exiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred in chat loop: {e}")
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    chat_main()

