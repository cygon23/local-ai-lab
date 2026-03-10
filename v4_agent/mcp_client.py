"""
mcp_client.py
-------------
Connects your agent to MCP (Model Context Protocol) servers.

WHAT IS MCP — THE REAL EXPLANATION:

  Before MCP, every AI app had to write custom integrations:
    - "I want my agent to use Google Drive" → write GoogleDriveTools class
    - "I want it to use GitHub" → write GitHubTools class
    - "I want it to use Slack" → write SlackTools class

  Each integration was one-off, incompatible with other agents.
  If you switched from LangChain to AutoGen, you rewrote everything.

  MCP solves this with a STANDARD PROTOCOL:
    - Tool providers write ONE MCP server
    - Any MCP-compatible agent can use it immediately
    - Tools self-describe via a standard schema
    - No custom integration code needed

  Think of MCP like USB:
    Before USB: every peripheral had its own connector
    After USB:  one standard, works everywhere
    MCP does the same for AI tools

THE MCP PROTOCOL — HOW IT WORKS:

  1. You connect to an MCP server (via stdio or HTTP/SSE)
  2. You call list_tools() → server returns available tools + schemas
  3. You call call_tool(name, args) → server executes and returns result
  4. That's it. The protocol is that simple.

  The magic: your agent doesn't need to know ANYTHING about the tool's
  implementation. It just reads the schema and calls by name.

MCP TRANSPORT TYPES:
  - stdio: Server runs as a subprocess, communicate via stdin/stdout
           Used for local tools (filesystem, code execution)
           Example: npx @modelcontextprotocol/server-filesystem

  - HTTP/SSE: Server runs as a web service, communicate via HTTP
              Used for remote tools (cloud services, APIs)
              Example: Claude.ai's Gmail, Google Drive connectors

IN THIS FILE:
  We implement an MCP client that:
  1. Connects to a stdio MCP server
  2. Lists its tools automatically
  3. Registers them in our tool registry
  4. Routes calls to the MCP server transparently

  The agent doesn't know or care that a tool comes from MCP vs local Python.
  The tool registry is the same. The ReAct loop is unchanged.

AVAILABLE FREE MCP SERVERS TO TRY:
  # Filesystem MCP server (Node.js required)
  npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir

  # Fetch/web MCP server
  npx -y @modelcontextprotocol/server-fetch

  # Memory MCP server
  npx -y @modelcontextprotocol/server-memory

  # SQLite MCP server
  npx -y @modelcontextprotocol/server-sqlite --db-path ./data/agent.db

SETUP:
  pip install mcp
  npm install -g @modelcontextprotocol/server-filesystem  (optional, to test)
"""

import json
import subprocess
import threading
import time
from typing import Any


class MCPStdioClient:
    """
    MCP client that communicates with a server via stdio (subprocess).

    HOW STDIO MCP WORKS:
      1. We launch the MCP server as a child process
      2. We write JSON-RPC messages to its stdin
      3. We read responses from its stdout
      4. The server runs in the background until we terminate it

    JSON-RPC is the message format MCP uses:
      Request:  {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
      Response: {"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}

    This is the same JSON-RPC used by VS Code's Language Server Protocol.
    If you've worked with LSPs, MCP will feel very familiar.
    """

    def __init__(self, command: list[str]):
        """
        command: the shell command to start the MCP server
        Example: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        """
        self.command = command
        self.process = None
        self._message_id = 0
        self._lock = threading.Lock()
        self.tools = []        # populated after connect()
        self.connected = False

    def connect(self) -> bool:
        """
        Start the MCP server subprocess and initialize the connection.
        Returns True if successful.

        THE INITIALIZATION HANDSHAKE:
          1. Client sends: initialize (with client info + capabilities)
          2. Server responds: initialized (with server info + capabilities)
          3. Client sends: notifications/initialized
          4. Now tools are available
        """
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,           # line-buffered
            )

            # Step 1: Send initialize
            init_response = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "local-ai-lab", "version": "4.0"}
            })

            if "error" in init_response:
                return False

            # Step 2: Send initialized notification
            self._send_notification("notifications/initialized")

            # Step 3: List available tools
            tools_response = self._send_request("tools/list", {})
            self.tools = tools_response.get("result", {}).get("tools", [])

            self.connected = True
            return True

        except FileNotFoundError:
            # Command not found (e.g., npx not installed)
            return False
        except Exception:
            return False

    def list_tools(self) -> list[dict]:
        """Return the list of tools available from this MCP server."""
        return self.tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool on the MCP server and return the result as a string.

        MCP tool results can be text, images, or embedded resources.
        We handle text results here — the most common case.
        """
        if not self.connected:
            return "Error: MCP server not connected."

        try:
            response = self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })

            if "error" in response:
                return f"MCP error: {response['error'].get('message', 'Unknown error')}"

            result = response.get("result", {})
            content = result.get("content", [])

            # Extract text from content blocks
            texts = []
            for block in content:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))

            return "\n".join(texts) if texts else str(result)

        except Exception as e:
            return f"Error calling MCP tool {tool_name}: {e}"

    def disconnect(self):
        """Terminate the MCP server subprocess."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
            self.connected = False

    def _next_id(self) -> int:
        with self._lock:
            self._message_id += 1
            return self._message_id

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        msg_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params
        }

        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            # Read response (blocking, with timeout via readline)
            for _ in range(50):  # max 5 seconds (50 × 100ms)
                line = self.process.stdout.readline()
                if line.strip():
                    return json.loads(line)
                time.sleep(0.1)

            return {"error": {"message": "Timeout waiting for MCP response"}}

        except Exception as e:
            return {"error": {"message": str(e)}}

    def _send_notification(self, method: str, params: dict = None):
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        try:
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()
        except Exception:
            pass


def mcp_tool_schema_to_local(mcp_tool: dict, server_name: str) -> dict:
    """
    Convert an MCP tool schema to our local TOOL_SCHEMAS format.

    MCP tool schema format:
    {
        "name": "read_file",
        "description": "Read the complete contents of a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        }
    }

    Our format:
    {
        "name": "mcp_filesystem_read_file",   ← prefixed to avoid collisions
        "description": "...",
        "parameters": {"path": "File path"},
        "returns": "Tool output from MCP server"
    }
    """
    mcp_name = mcp_tool.get("name", "unknown")
    # Prefix with server name to avoid collisions with local tools
    local_name = f"mcp_{server_name}_{mcp_name}"

    # Extract parameter descriptions
    input_schema = mcp_tool.get("inputSchema", {})
    properties = input_schema.get("properties", {})
    parameters = {
        k: v.get("description", f"{k} parameter")
        for k, v in properties.items()
    }

    return {
        "name": local_name,
        "description": f"[MCP:{server_name}] {mcp_tool.get('description', '')}",
        "parameters": parameters,
        "returns": f"Result from MCP {server_name} server.",
        "_mcp_original_name": mcp_name,      # keep original name for routing
        "_mcp_server": server_name,
    }


# Global registry of active MCP clients
# Key: server_name, Value: MCPStdioClient instance
_active_mcp_clients: dict[str, MCPStdioClient] = {}


def connect_mcp_server(server_name: str, command: list[str]) -> tuple[bool, list[dict]]:
    """
    Connect to an MCP server and return its tools.

    Returns: (success, list_of_local_tool_schemas)

    Usage:
      success, tools = connect_mcp_server(
          "filesystem",
          ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./data/workspace"]
      )
      if success:
          # Register tools in TOOL_FUNCTIONS and TOOL_SCHEMAS
          for tool in tools:
              TOOL_SCHEMAS.append(tool)
              TOOL_FUNCTIONS[tool["name"]] = make_mcp_caller(tool)
    """
    client = MCPStdioClient(command)
    if not client.connect():
        return False, []

    _active_mcp_clients[server_name] = client
    local_tools = [
        mcp_tool_schema_to_local(t, server_name)
        for t in client.list_tools()
    ]
    return True, local_tools


def call_mcp_tool(server_name: str, original_tool_name: str, args: dict) -> str:
    """Route a tool call to the correct MCP server."""
    client = _active_mcp_clients.get(server_name)
    if not client:
        return f"Error: MCP server '{server_name}' not connected."
    return client.call_tool(original_tool_name, args)


def disconnect_all():
    """Clean up all MCP server connections on shutdown."""
    for client in _active_mcp_clients.values():
        client.disconnect()
    _active_mcp_clients.clear()


def get_connected_servers() -> list[str]:
    """List names of currently connected MCP servers."""
    return [name for name, c in _active_mcp_clients.items() if c.connected]