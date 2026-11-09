import logfire
from typing import Dict, Any, Callable, Optional, List
from pydantic import BaseModel, Field
from app.services.tools.calculator import SafeCalculator
from app.services.tools.python_executor import RestrictedPythonExecutor
from app.services.tools.datetime_util import DateTimeUtil


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = Field(default_factory=list)


class ToolRegistry:
    """
    Centralized, secure tool registry for the Tool Agent.
    Manages registration, input validation, execution, and error handling.
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable, required: Optional[List[str]] = None):
        """Register a new tool callable with schema definition."""
        self._tools[name] = func
        self._schemas[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            required=required or list(parameters.keys())
        )

    def _register_default_tools(self):
        """Register initial built-in utilities."""
        # 1. Calculator
        self.register(
            name="calculator",
            description="Safely evaluates mathematical expressions, scientific formulas, percentages, and powers.",
            parameters={"expression": {"type": "string", "description": "Arithmetic or math formula (e.g., '(450 * 1.15) / 12')"}},
            required=["expression"],
            func=lambda expression: SafeCalculator.evaluate(expression)
        )

        # 2. Python Executor (Restricted)
        self.register(
            name="python_executor",
            description="Executes restricted Python code for data processing, statistical calculations, and list transformations.",
            parameters={"code": {"type": "string", "description": "Python snippet assigning calculated results to local variables"}},
            required=["code"],
            func=lambda code: RestrictedPythonExecutor.execute(code)
        )

        # 3. DateTime Utilities
        self.register(
            name="datetime_util",
            description="Retrieves current date/time, day of the week, or calculates date intervals across timezones.",
            parameters={
                "action": {"type": "string", "description": "'now' or 'delta'"},
                "timezone": {"type": "string", "description": "Timezone name (e.g., 'UTC', 'US/Eastern', 'Asia/Kolkata')"},
                "days": {"type": "integer", "description": "Days to add/subtract (for 'delta' action)"}
            },
            required=["action"],
            func=lambda action, timezone="UTC", days=0: (
                DateTimeUtil.get_current_datetime(timezone)
                if action == "now"
                else DateTimeUtil.calculate_date_delta(days=days)
            )
        )

    def get_tool_schema(self, name: str) -> Optional[ToolDefinition]:
        return self._schemas.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [schema.model_dump() for schema in self._schemas.values()]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool with argument validation and error containment."""
        if tool_name not in self._tools:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}"
            }

        schema = self._schemas[tool_name]
        missing = [param for param in schema.required if param not in arguments]
        if missing:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Missing required parameter(s): {', '.join(missing)}"
            }

        with logfire.span("Tool Execution", tool=tool_name, arguments=arguments):
            try:
                func = self._tools[tool_name]
                result = func(**arguments)
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result
                }
            except Exception as e:
                return {
                    "success": False,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "error": f"ExecutionError: {str(e)}"
                }


# Global tool registry singleton
tool_registry = ToolRegistry()

