from mcp.server import FastMCP

mcp = FastMCP("workflow", host="127.0.0.1", port=8000)

@mcp.prompt()
def workflow() -> str:
    """Gives the math workflow"""
    return "Calculate 3+3 and multiply result with 2"

if __name__ == '__main__':
    mcp.run(transport="streamable-http")
