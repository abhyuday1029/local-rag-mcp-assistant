import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


BASE_DIR = Path(__file__).resolve().parent


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "weather_server.py")],
    )

    print("Starting weather server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("MCP connection established.")

            tools = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools.tools])

            result = await session.call_tool(
                "get_weather",
                arguments={"city": "Mumbai"},
            )

            print("Tool reported an error:", result.isError)

            for block in result.content:
                if block.type == "text":
                    print("Result:", block.text)


if __name__ == "__main__":
    asyncio.run(main())