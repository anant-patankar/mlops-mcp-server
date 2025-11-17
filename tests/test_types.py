from mlops_mcp._types import err


def test_err_shape():
    result = err("something broke")
    assert result["success"] is False
    assert result["error"] == "something broke"


def test_err_empty_message():
    # edge case — empty string is still a valid error
    result = err("")
    assert result["success"] is False
    assert result["error"] == ""


def test_err_does_not_bleed_extra_keys():
    result = err("oops")
    assert set(result.keys()) == {"success", "error"}