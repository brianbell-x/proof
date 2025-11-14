"""
This module implements a chat agent that can use the ProverTool for claim verification.

The ProverAgent class has been moved to tools/prover_tool.py to consolidate
all prover-related logic in one place.
"""

import os
import json
import logging
import time
from typing import Dict, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Import ProverAgent and ProverTool from prover_tool module
# All ProverAgent logic is now in tools/prover_tool.py
from tools.prover_tool import ProverAgent, ProverTool, get_tool_schema as get_prover_schema

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file or environment variables.")

# Set up logging for ChatAgent
logger = logging.getLogger(__name__)


class ChatAgent:
    """
    Chat agent that uses grok-4-fast for general conversation.
    Has access to the ProverTool for claim verification when needed.
    Maintains full conversation history to answer follow-up questions.
    """

    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        """
        Initialize the Chat Agent.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use (default: grok-4-fast)
        """
        self.api_key = api_key
        self.model = model
        # Initialize OpenAI client for chat interactions
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Initialize the ProverTool - this is the only tool the chat agent has access to
        # The chat agent does NOT have direct access to web_search or python_execute tools
        self.prover_tool = ProverTool(api_key, model)
        
        # System prompt explaining that the chat agent has access to the prover tool
        # This tells the agent when and how to use the prove_claim tool
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = f"""Current date: {current_date}

You are a helpful chat assistant. You have access to a claim verification tool called 'prove_claim' that can rigorously analyze whether claims are true or false.

When to use the prove_claim tool:
- When the user asks if a claim is true or false
- When the user wants to verify a statement
- When the user asks "is it true that..." or similar verification questions
- When the user presents a claim and wants to know if it's accurate

The prove_claim tool will return a verdict (PROVEN, DISPROVEN, UNSUPPORTED, or UNVERIFIABLE) along with detailed reasoning, evidence, and derivation steps. Use this information to provide a clear, helpful answer to the user.

For general conversation and questions that don't involve claim verification, respond directly without using tools.
"""
        
        # Only include the prover tool schema - NOT web_search or python_execute
        self.tool_schemas = [get_prover_schema()]
        
        # Maintain conversation history - all model responses are stored here
        # This allows the agent to answer follow-up questions with full context
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        This method maintains full conversation history by appending all messages
        (user, assistant, and tool responses) to self.messages. This ensures
        the agent has complete context for follow-up questions.
        
        Args:
            user_message: The user's message/query
            
        Returns:
            Dict containing the assistant's response and metadata
        """
        start_time = time.time()
        
        # Add the user's message to the conversation history
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Track token usage and costs
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # Main conversation loop - handles tool calls if needed
        while True:
            try:
                # Make API call to the chat model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tool_schemas,
                    tool_choice="auto"  # Let the model decide when to use tools
                )
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    total_prompt_tokens += getattr(usage, 'prompt_tokens', 0)
                    total_completion_tokens += getattr(usage, 'completion_tokens', 0)
                
                # Extract the assistant's message
                message = response.choices[0].message
                message_dict = message.model_dump() if hasattr(message, 'model_dump') else message
                
                # Add the assistant's response to conversation history
                # This ensures all model responses are available for follow-up questions
                self.messages.append(message_dict)
                
                # Check if the model wants to use a tool
                tool_calls = getattr(message, 'tool_calls', None)
                if not tool_calls and isinstance(message_dict, dict):
                    tool_calls = message_dict.get('tool_calls')
                
                # If no tool calls, we're done - return the final response
                if not tool_calls:
                    content = message.content or ""
                    
                    # Calculate costs
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
                
                # Process tool calls
                logger.info(f"Processing {len(tool_calls)} tool calls")
                
                for tool_call in tool_calls:
                    # Extract tool call information
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
                    
                    # Execute the tool (only prove_claim is available)
                    if tool_name == "prove_claim":
                        logger.info(f"[TOOL CALL] prove_claim with claim: {arguments.get('claim', '')[:50]}...")
                        tool_result = self.prover_tool.prove_claim(
                            claim=arguments.get('claim', ''),
                            max_iterations=arguments.get('max_iterations')
                        )
                    else:
                        tool_result = {
                            "error": f"Unknown tool: {tool_name}",
                            "tool_call_id": tool_call_id
                        }
                    
                    # Add tool result to conversation history
                    # This ensures the full prover analysis is available for follow-up questions
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result)
                    }
                    self.messages.append(tool_message)
                    logger.info(f"[TOOL RESULT] Added to conversation history")
                
                # Continue the loop to let the model process the tool results
                # The model will generate a final response incorporating the tool results
                
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
        """
        Calculate API costs based on token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Dict with cost breakdown
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

    def reset_conversation(self):
        """
        Reset the conversation history.
        Useful for starting a new conversation thread.
        """
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        logger.info("Conversation history reset")


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


def chat_main():
    """
    Main function for the Chat Agent.
    Provides an interactive chat interface where users can ask questions
    and verify claims. The agent will automatically use the prove_claim tool
    when appropriate.
    """
    print("Chat Agent with Claim Verification")
    print("===================================")
    print("Ask me anything! I can verify claims using rigorous analysis.")
    print("Type 'quit' to exit, 'reset' to start a new conversation.")
    print("Set PROVER_VERBOSE=1 to see detailed logs in console.")
    
    # Initialize the chat agent
    chat_agent = ChatAgent(api_key)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                logger.info("User requested exit")
                break
            
            if user_input.lower() == 'reset':
                chat_agent.reset_conversation()
                print("Conversation reset. Starting fresh!")
                continue
            
            if not user_input:
                continue
            
            logger.info(f"User message: {user_input}")
            
            # Get response from chat agent
            result = chat_agent.chat(user_input)
            
            # Display response
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
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
            print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    # Run ChatAgent by default for conversational interface
    # To use ProverAgent directly instead, change this to: main()
    chat_main()