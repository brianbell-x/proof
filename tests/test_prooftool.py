import os
import sys
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tool.proof_tool import ProofAgent
from dotenv import load_dotenv

load_dotenv()

# ANSI color codes for colored borders
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Border colors
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    
    # Text colors
    WHITE = '\033[97m'
    GRAY = '\033[90m'


def print_bordered_section(title: str, content: str, color: str = Colors.CYAN, width: int = 80):
    border_char = '═'
    corner_char = '╔'
    corner_end = '╗'
    side_char = '║'
    corner_bottom_start = '╚'
    corner_bottom_end = '╝'
    
    # Calculate width
    title_width = len(title) + 2
    actual_width = max(width, title_width + 4)
    
    # Top border with title
    top_border = f"{color}{corner_char}{border_char * (actual_width - 2)}{corner_end}{Colors.RESET}"
    title_line = f"{color}{side_char}{Colors.RESET} {Colors.BOLD}{Colors.WHITE}{title}{Colors.RESET} {color}{side_char}{Colors.RESET}"
    
    # Content lines
    content_lines = content.split('\n')
    formatted_lines = []
    for line in content_lines:
        # Truncate very long lines
        if len(line) > actual_width - 4:
            line = line[:actual_width - 7] + "..."
        formatted_lines.append(f"{color}{side_char}{Colors.RESET} {line:<{actual_width - 4}} {color}{side_char}{Colors.RESET}")
    
    # Bottom border
    bottom_border = f"{color}{corner_bottom_start}{border_char * (actual_width - 2)}{corner_bottom_end}{Colors.RESET}"
    
    print()
    print(top_border)
    print(title_line)
    print(f"{color}{side_char}{' ' * (actual_width - 2)}{side_char}{Colors.RESET}")
    for line in formatted_lines:
        print(line)
    print(f"{color}{side_char}{' ' * (actual_width - 2)}{side_char}{Colors.RESET}")
    print(bottom_border)
    print()


class ProofToolTester:
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.api_key = api_key
        self.model = model
        self.proof_agent = ProofAgent(api_key, model)
        
        # Track information for display
        self.input_claim = None
        self.system_message = None
        self.tool_calls_history: List[Dict[str, Any]] = []
        self.tool_results_history: List[Dict[str, Any]] = []
        self.model_outputs: List[Dict[str, Any]] = []
        self.final_result: Optional[Dict[str, Any]] = None
        
        # Override the _execute_tool method to capture tool calls
        self._original_execute_tool = self.proof_agent._execute_tool
        self.proof_agent._execute_tool = self._capture_tool_execution
    
    def _capture_tool_execution(self, tool_call):
        # Extract tool call info
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
        elif isinstance(tool_call, dict):
            tool_name = tool_call.get('function', {}).get('name')
            arguments = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
            tool_call_id = tool_call.get('id')
        else:
            return self._original_execute_tool(tool_call)
        
        # Store tool call
        tool_call_info = {
            "id": tool_call_id,
            "name": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat()
        }
        self.tool_calls_history.append(tool_call_info)
        
        # Display tool call
        tool_call_display = f"Tool: {Colors.BOLD}{Colors.YELLOW}{tool_name}{Colors.RESET}\n"
        tool_call_display += f"ID: {tool_call_id}\n"
        tool_call_display += f"Arguments:\n{json.dumps(arguments, indent=2)}"
        print_bordered_section("🔧 TOOL CALL", tool_call_display, Colors.YELLOW)
        
        # Execute tool
        result = self._original_execute_tool(tool_call)
        
        # Store tool result
        self.tool_results_history.append({
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display tool result
        result_display = f"Tool: {Colors.BOLD}{Colors.GREEN}{tool_name}{Colors.RESET}\n"
        result_display += f"ID: {tool_call_id}\n"
        
        # Format result nicely
        if isinstance(result, dict):
            if "error" in result:
                result_display += f"{Colors.RED}Error: {result['error']}{Colors.RESET}\n"
            else:
                result_str = json.dumps(result, indent=2)
                # Truncate very long results
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "\n... (truncated)"
                result_display += f"Result:\n{result_str}"
        else:
            result_display += f"Result: {str(result)}"
        
        print_bordered_section("📤 TOOL RESULT", result_display, Colors.GREEN)
        
        return result
    
    def test_claim(self, claim: str, max_iterations: int = None):
        import time
        from tools.proof_tool import _strip_markdown_code_fences
        
        self.input_claim = claim
        self.system_message = self.proof_agent.system_prompt
        self.tool_calls_history = []
        self.tool_results_history = []
        self.model_outputs = []
        
        # Display input
        print_bordered_section("INPUT", claim, Colors.BLUE)
        
        # Display system message
        system_display = self.system_message
        print_bordered_section("SYSTEM MESSAGE", system_display, Colors.MAGENTA)
        
        # Replicate prove_claim logic but with display hooks
        start_time = time.time()
        messages = [
            {"role": "system", "content": self.proof_agent.system_prompt},
            {"role": "user", "content": claim}
        ]
        
        iteration = 0
        final_result = None
        tools_used = set()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        while True:
            iteration += 1
            try:
                response = self.proof_agent.client.chat.completions.create(
                    model=self.proof_agent.model,
                    messages=messages,
                    tools=self.proof_agent.tool_schemas,
                    tool_choice="auto"
                )
                
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    total_prompt_tokens += getattr(usage, 'prompt_tokens', 0)
                    total_completion_tokens += getattr(usage, 'completion_tokens', 0)
                
                message = response.choices[0].message
                message_dict = message.model_dump() if hasattr(message, 'model_dump') else message
                messages.append(message_dict)
                
                content = message.content or ""
                tool_calls = getattr(message, 'tool_calls', None)
                if not tool_calls and isinstance(message_dict, dict):
                    tool_calls = message_dict.get('tool_calls')
                
                finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                
                # Store and display model output
                model_output = {
                    "iteration": iteration,
                    "content": content,
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason
                }
                self.model_outputs.append(model_output)
                
                # Display model output
                output_display = f"Iteration: {iteration}\n"
                output_display += f"Finish Reason: {finish_reason}\n\n"
                
                if content:
                    # Try to parse as JSON for better display
                    try:
                        cleaned = _strip_markdown_code_fences(content)
                        parsed = json.loads(cleaned) if cleaned else {}
                        output_display += f"Content (JSON):\n{json.dumps(parsed, indent=2)}"
                    except:
                        # Truncate very long content
                        display_content = content
                        if len(display_content) > 1000:
                            display_content = display_content[:1000] + "\n... (truncated)"
                        output_display += f"Content:\n{display_content}"
                
                if tool_calls:
                    output_display += f"\n\n{Colors.YELLOW}Tool Calls Requested: {len(tool_calls)}{Colors.RESET}"
                
                print_bordered_section(f"💭 MODEL OUTPUT (Iteration {iteration})", output_display, Colors.CYAN)
                
                # Try to parse JSON response
                try:
                    cleaned_content = _strip_markdown_code_fences(content)
                    result = json.loads(cleaned_content) if cleaned_content else {}
                except:
                    result = {"error": "Invalid JSON response", "raw_content": content}
                
                # Check if verdict reached
                if "verdict" in result:
                    final_result = result
                    break
                
                # Process tool calls
                if tool_calls:
                    for tool_call in tool_calls:
                        if hasattr(tool_call, 'function'):
                            tool_name = tool_call.function.name
                            tool_call_id = tool_call.id
                        elif isinstance(tool_call, dict):
                            tool_name = tool_call.get('function', {}).get('name')
                            tool_call_id = tool_call.get('id')
                        else:
                            continue
                        
                        tools_used.add(tool_name)
                        tool_result = self.proof_agent._execute_tool(tool_call)
                        
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(tool_result)
                        }
                        messages.append(tool_message)
                
                if max_iterations and iteration >= max_iterations:
                    break
                    
            except Exception as e:
                end_time = time.time()
                elapsed_time = end_time - start_time
                cost_info = self.proof_agent._calculate_costs(total_prompt_tokens, total_completion_tokens)
                result = {
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
                self.final_result = result
                print_bordered_section("❌ ERROR", f"Error occurred: {str(e)}", Colors.RED)
                return result
        
        # Calculate final stats
        end_time = time.time()
        elapsed_time = end_time - start_time
        cost_info = self.proof_agent._calculate_costs(total_prompt_tokens, total_completion_tokens)
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
        else:
            final_result = {
                "error": "No verdict reached",
                "claim": claim,
                "iterations_completed": iteration,
                "tools_used": list(tools_used),
                "time_seconds": round(elapsed_time, 3),
                "tokens": tokens_info,
                "cost": cost_info
            }
        
        self.final_result = final_result
        
        # Display final result
        final_display = ""
        if "verdict" in final_result:
            verdict_color = Colors.GREEN if final_result["verdict"] == "PROVEN" else (
                Colors.RED if final_result["verdict"] == "DISPROVEN" else Colors.YELLOW
            )
            final_display += f"{Colors.BOLD}Verdict: {verdict_color}{final_result['verdict']}{Colors.RESET}\n\n"
        
        if "reasoning_summary" in final_result:
            final_display += f"Reasoning: {final_result['reasoning_summary']}\n\n"
        
        if "iterations_used" in final_result:
            final_display += f"Iterations: {final_result['iterations_used']}\n"
        
        if "time_seconds" in final_result:
            final_display += f"Time: {final_result['time_seconds']}s\n"
        
        if "cost" in final_result:
            cost = final_result["cost"]
            final_display += f"Cost: ${cost.get('total_usd', 0):.6f} (input: ${cost.get('input_usd', 0):.6f}, output: ${cost.get('output_usd', 0):.6f})\n"
        
        final_display += f"\n{Colors.GRAY}Full Result:{Colors.RESET}\n{json.dumps(final_result, indent=2)}"
        
        print_bordered_section("✅ FINAL RESULT", final_display, Colors.GREEN)
        
        return final_result


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(f"{Colors.RED}Error: OPENROUTER_API_KEY not found in environment variables.{Colors.RESET}")
        sys.exit(1)
    
    # Get claim from command line or use default
    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:])
    else:
        claim = input(f"{Colors.CYAN}Enter a claim to test: {Colors.RESET}").strip()
        if not claim:
            claim = "Water boils at 100°C at standard atmospheric pressure."
    
    # Get max iterations if provided
    max_iterations = None
    if len(sys.argv) > 2:
        try:
            max_iterations = int(sys.argv[-1])
        except ValueError:
            pass
    
    # Create tester and run
    tester = ProofToolTester(api_key)
    tester.test_claim(claim, max_iterations)


if __name__ == "__main__":
    main()

