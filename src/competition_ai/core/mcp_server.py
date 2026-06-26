from typing import Dict, Any
from competition_ai.core.architecture import CompetitorArchitecture

class CompetitionMCPServer:
    """
    MCP-compliant server interface for the Competitor Architecture.
    Allows external agents/tools to interact with the architecture.
    """
    def __init__(self, architecture: CompetitorArchitecture):
        self.arch = architecture

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches tool calls from an MCP client."""
        if tool_name == "solve_task":
            return {"result": self.arch.solve_arc(arguments["task"])}
        elif tool_name == "get_status":
            competition_id = arguments.get("competition_id", "")
            return self.arch.skills["monitoring"].get_status(competition_id)
        else:
            return {"error": f"Tool {tool_name} not found"}
