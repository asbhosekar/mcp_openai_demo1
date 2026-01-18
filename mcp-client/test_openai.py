"""Non-interactive test for MCP OpenAI client"""
import asyncio
import os
import sys
import json
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

def mcp_content_to_json(content):
    if content is None:
        return None
    if isinstance(content, list):
        return [mcp_content_to_json(c) for c in content]
    if hasattr(content, "text"):
        return content.text
    if isinstance(content, dict):
        return content
    return str(content)

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
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        })
    return openai_tools

async def test_query(query: str):
    """Run a single test query"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found")
        return

    openai_client = OpenAI(api_key=openai_api_key)
    exit_stack = AsyncExitStack()

    # Get the path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, "..", "server", "main.py")

    server_params = StdioServerParameters(
        command="python",
        args=[server_path]
    )

    async with exit_stack:
        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize()

        # List tools
        response = await session.list_tools()
        tools = response.tools
        print("=" * 50)
        print("CONNECTED TO SERVER")
        print(f"Available tools: {[t.name for t in tools]}")
        print("=" * 50)

        openai_tools = convert_mcp_tools_to_openai(tools)

        # Process query
        print(f"\nQuery: {query}")
        print("-" * 50)

        messages = [{"role": "user", "content": query}]

        while True:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls,
            })

            if not assistant_message.tool_calls:
                print(f"\nAI Response: {assistant_message.content}")
                break

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"\n[AI calling tool: {tool_name}({tool_args})]")

                try:
                    result = await session.call_tool(tool_name, tool_args)
                    tool_output = mcp_content_to_json(result.content)
                    print(f"[Tool output: {tool_output}]")
                except Exception as e:
                    tool_output = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(tool_output),
                })

        print("=" * 50)
        print("TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    query = "Create a folder called 'asb', then create a file named 'welcome.txt' inside it with the text 'Hello Ashish, Welcome', and finally show the contents of the file"
    asyncio.run(test_query(query))
