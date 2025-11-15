from typing import Any


def err(msg: str) -> dict[str, Any]:
    return {"success": False, "error": msg}
