import os
import json
import logging
import time
from typing import Dict, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tools.proof_tool import ProofTool, get_tool_schema as get_proof_schema

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file or environment variables.")

logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.proof_tool = ProofTool(api_key, model)
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = self._load_prompt("prompts/chat_agent_prompt.md").format(current_date=current_date)
        self.tool_schemas = [get_proof_schema()]
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    def _load_prompt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {path}")
            return "You are a helpful assistant."

    def chat(self, user_message: str) -> Dict[str, Any]:
        start_time = time.time()

        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tool_schemas,
                    tool_choice="auto"
                )
                
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    total_prompt_tokens += getattr(usage, 'prompt_tokens', 0)
                    total_completion_tokens += getattr(usage, 'completion_tokens', 0)
                
                message = response.choices[0].message
                message_dict = message.model_dump() if hasattr(message, 'model_dump') else message
                self.messages.append(message_dict)
                
                tool_calls = getattr(message, 'tool_calls', None)
                if not tool_calls and isinstance(message_dict, dict):
                    tool_calls = message_dict.get('tool_calls')
                
                if not tool_calls:
                    content = message.content or ""
                    cost_info = self._calculate_costs(total_prompt_tokens, total_completion_tokens)
                    elapsed_time = time.time() - start_time
                    return {
                        "response": content,
                        "time_seconds": round(elapsed_time, 3),
                        "tokens": {
                            "prompt": total_prompt_tokens,
                            "completion": total_completion_tokens,
                            "total": total_prompt_tokens + total_completion_tokens
                        },
                        "cost": cost_info
                    }
                
                logger.info(f"Processing {len(tool_calls)} tool calls")
                
                for tool_call in tool_calls:
                    if hasattr(tool_call, 'function'):
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                    elif isinstance(tool_call, dict):
                        tool_name = tool_call.get('function', {}).get('name')
                        arguments = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                        tool_call_id = tool_call.get('id')
                    else:
                        logger.warning(f"Unexpected tool_call format: {type(tool_call)}")
                        continue
                    
                    if tool_name == "prove_claim":
                        logger.info(f"[TOOL CALL] prove_claim with claim: {arguments.get('claim', '')[:50]}...")
                        tool_result = self.proof_tool.prove_claim(
                            claim=arguments.get('claim', ''),
                            max_iterations=arguments.get('max_iterations')
                        )
                        if "tokens" in tool_result:
                            proof_tokens = tool_result["tokens"]
                            total_prompt_tokens += proof_tokens.get("prompt", 0)
                            total_completion_tokens += proof_tokens.get("completion", 0)
                            logger.info(f"[COST TRACKING] Added proof tokens: prompt={proof_tokens.get('prompt', 0)}, completion={proof_tokens.get('completion', 0)}")
                    else:
                        tool_result = {
                            "error": f"Unknown tool: {tool_name}",
                            "tool_call_id": tool_call_id
                        }
                    
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result)
                    }
                    self.messages.append(tool_message)
                    logger.info(f"[TOOL RESULT] Added to conversation history")
                
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                elapsed_time = time.time() - start_time
                cost_info = self._calculate_costs(total_prompt_tokens, total_completion_tokens)
                return {
                    "error": str(e),
                    "time_seconds": round(elapsed_time, 3),
                    "tokens": {
                        "prompt": total_prompt_tokens,
                        "completion": total_completion_tokens,
                        "total": total_prompt_tokens + total_completion_tokens
                    },
                    "cost": cost_info
                }

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

    def reset_conversation(self):
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        logger.info("Conversation history reset")

def chat_main():
    print("Type 'quit' to exit, 'reset' to start a new conversation.")
    chat_agent = ChatAgent(api_key)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                logger.info("User requested exit")
                break
            
            if user_input.lower() == 'reset':
                chat_agent.reset_conversation()
                print("Conversation reset. Starting fresh!")
                continue
            
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
                response = result.get("response", "No response generated")
                time_seconds = result.get("time_seconds", 0)
                cost_info = result.get("cost", {})
                print(f"\nAssistant: {response}")
                print(f"   Time: {time_seconds}s")
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