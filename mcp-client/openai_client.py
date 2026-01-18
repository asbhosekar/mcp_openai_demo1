import asyncio
import os
from pyexpat.errors import messages
import sys
import json
import time
from typing import Optional
from contextlib import AsyncExitStack

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI

load_dotenv()

def mcp_content_to_json(content):
    """
    Converts MCP tool response content into JSON-serializable format.
    """
    if content is None:
        return None

    # MCP usually returns a list
    if isinstance(content, list):
        return [mcp_content_to_json(c) for c in content]

    # TextContent (most common)
    if hasattr(content, "text"):
        return content.text

    # Fallback for dict-like
    if isinstance(content, dict):
        return content

    # Last resort
    return str(content)


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        command = "python" if server_script_path.endswith(".py") else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path])

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools

        print("\nConnected to server with tools:", [t.name for t in tools])

        self.tools = convert_mcp_tools_to_openai(tools)

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]

        while True:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message

            # 1️⃣ Always append assistant message FIRST
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls,
                }
            )

            # 2️⃣ If no tool calls → we are DONE
            if not assistant_message.tool_calls:
                return assistant_message.content

            # 3️⃣ Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"\n[GPT requested tool call: {tool_name} with args {tool_args}]")

                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_output = mcp_content_to_json(result.content)
                except Exception as e:
                    tool_output = {"error": str(e)}

                # 4️⃣ Append tool response IMMEDIATELY after assistant tool_call
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_output),
                    }
                )

    async def chat_loop(self):
        print("\nMCP Client Started! Type 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break

            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        await self.exit_stack.aclose()


def clean_schema(schema):
    if isinstance(schema, dict):
        schema.pop("title", None)
        if "properties" in schema:
            for k, v in schema["properties"].items():
                schema["properties"][k] = clean_schema(v)
    return schema


def convert_mcp_tools_to_openai(mcp_tools):
    openai_tools = []

    for tool in mcp_tools:
        parameters = clean_schema(tool.inputSchema)

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )

    return openai_tools


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
