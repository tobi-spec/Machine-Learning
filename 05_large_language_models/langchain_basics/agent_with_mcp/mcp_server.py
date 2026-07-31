from mcp.server import FastMCP

mcp = FastMCP("math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b


if __name__ == '__main__':
    mcp.run()