from mlops_mcp.server import create_server
from fastmcp import FastMCP


def test_create_server_returns_fastmcp():
    app = create_server()
    assert isinstance(app, FastMCP)


def test_server_has_name():
    app = create_server()
    assert app.name == "mlops-mcp-server"


def test_create_server_does_not_raise():
    # smoke test — if something in registration is broken, this catches it
    try:
        create_server()
    except Exception as exc:
        pytest.fail(f"create_server() raised unexpectedly: {exc}")