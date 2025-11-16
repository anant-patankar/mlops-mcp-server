from fastmcp import FastMCP
from .router import register_tier_one_tools

def create_server() -> FastMCP:
    app = FastMCP(name="mlops-mcp-server")
    register_tier_one_tools(app)
    return app

def main() -> None:
    app = create_server()
    app.run()