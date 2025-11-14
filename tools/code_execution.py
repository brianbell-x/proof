import io
import contextlib
import traceback
from typing import Dict, Any
from datetime import datetime


class CodeExecutionTool:
    def __init__(self, timeout: int = 10, max_output_length: int = None):
        self.timeout = timeout
        self.max_output_length = max_output_length
        self._globals = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "statistics": __import__("statistics"),
            "datetime": __import__("datetime"),
            "json": __import__("json"),
        }

    def execute(self, code: str) -> Dict[str, Any]:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = {
            "code": code,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "output": "",
            "error": "",
            "execution_time": None
        }

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                start_time = datetime.now()
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, self._globals)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

            result["success"] = True
            result["output"] = stdout_capture.getvalue()
            result["execution_time"] = execution_time

            stderr_output = stderr_capture.getvalue()
            if stderr_output:
                result["warnings"] = stderr_output

        except (SyntaxError, NameError, TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            result["output"] = stdout_capture.getvalue()

        if self.max_output_length and len(result["output"]) > self.max_output_length:
            result["output"] = result["output"][:self.max_output_length] + "... (truncated)"
            result["truncated"] = True

        return result

    def calculate(self, expression: str) -> Dict[str, Any]:
        code = f"""
result = {expression}
print(repr(result))
"""
        return self.execute(code)

    def analyze_data(self, data_code: str) -> Dict[str, Any]:
        return self.execute(data_code)


def get_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Execute Python code for calculations, data analysis, and mathematical verification. Use this for numerical computations, statistical analysis, or when you need to perform calculations to verify claims.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Include print statements to show results. Available modules: math, statistics, datetime, json."
                    }
                },
                "required": ["code"]
            }
        }
    }


if __name__ == "__main__":
    tool = CodeExecutionTool()

    result = tool.calculate("2 * 3.14159 * 5")
    print("Calculation result:")
    print(result)

    code_result = tool.execute("""
import math
radius = 5
area = math.pi * radius ** 2
volume = (4/3) * math.pi * radius ** 3
print(f"Area: {area:.2f}")
print(f"Volume: {volume:.2f}")
""")
    print("\nCode execution result:")
    print(code_result)

