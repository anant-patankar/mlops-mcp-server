import time
import pytest
import yaml

from mlops_mcp.pipelines import (
    _build_stage_graph,
    _get_stages,
    _has_cycle,
    add_pipeline_stage,
    create_pipeline,
    get_pipeline,
    get_pipeline_status,
    list_pipelines,
    remove_pipeline_stage,
    validate_pipeline,
    visualize_pipeline,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

MINIMAL_STAGE = {"cmd": "python train.py", "deps": [], "outs": []}


def write_pipeline(path, stages):
    path.write_text(yaml.safe_dump({"stages": stages}))
    return str(path)


# ---------------------------------------------------------------------------
# _get_stages
# ---------------------------------------------------------------------------

def test_get_stages_returns_stages_from_dict():
    assert _get_stages({"stages": {"a": {}}}) == {"a": {}}


def test_get_stages_missing_key_returns_empty():
    assert _get_stages({}) == {}


def test_get_stages_non_dict_payload_returns_empty():
    assert _get_stages(None) == {}
    assert _get_stages("bad") == {}


# ---------------------------------------------------------------------------
# _build_stage_graph
# ---------------------------------------------------------------------------

def test_build_stage_graph_linear_chain():
    stages = {
        "prep": {"outs": ["data.csv"], "deps": []},
        "train": {"outs": ["model.pkl"], "deps": ["data.csv"]},
    }
    graph = _build_stage_graph(stages)
    assert graph["prep"] == {"train"}
    assert graph["train"] == set()


def test_build_stage_graph_no_edges_when_no_shared_artifacts():
    stages = {
        "a": {"outs": ["x.csv"], "deps": []},
        "b": {"outs": ["y.csv"], "deps": []},
    }
    graph = _build_stage_graph(stages)
    assert graph == {"a": set(), "b": set()}


def test_build_stage_graph_ignores_external_deps():
    # dep not produced by any stage — should not create a graph edge
    stages = {"train": {"outs": ["model.pkl"], "deps": ["external.csv"]}}
    graph = _build_stage_graph(stages)
    assert graph == {"train": set()}


# ---------------------------------------------------------------------------
# _has_cycle
# ---------------------------------------------------------------------------

def test_has_cycle_detects_simple_cycle():
    graph = {"a": {"b"}, "b": {"a"}}
    assert _has_cycle(graph) is True


def test_has_cycle_linear_dag_no_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": set()}
    assert _has_cycle(graph) is False


def test_has_cycle_empty_graph():
    assert _has_cycle({}) is False


def test_has_cycle_three_node_cycle():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    assert _has_cycle(graph) is True


def test_has_cycle_diamond_dag_no_cycle():
    graph = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": set()}
    assert _has_cycle(graph) is False


# ---------------------------------------------------------------------------
# create_pipeline
# ---------------------------------------------------------------------------

def test_create_pipeline_writes_yaml(tmp_path):
    p = str(tmp_path / "test.pipeline.yaml")
    result = create_pipeline(p, {"train": MINIMAL_STAGE})
    assert result["success"] is True
    loaded = yaml.safe_load(open(p).read())
    assert "stages" in loaded
    assert "train" in loaded["stages"]


def test_create_pipeline_empty_stages_fails():
    result = create_pipeline("/tmp/x.yaml", {})
    assert result["success"] is False
    assert "empty" in result["error"]


def test_create_pipeline_creates_parent_dirs(tmp_path):
    p = str(tmp_path / "nested" / "deep" / "pipeline.yaml")
    result = create_pipeline(p, {"train": MINIMAL_STAGE})
    assert result["success"] is True


# ---------------------------------------------------------------------------
# get_pipeline
# ---------------------------------------------------------------------------

def test_get_pipeline_returns_stage_count(tmp_path):
    p = tmp_path / "p.pipeline.yaml"
    write_pipeline(p, {"a": MINIMAL_STAGE, "b": MINIMAL_STAGE})
    result = get_pipeline(str(p))
    assert result["success"] is True
    assert result["stage_count"] == 2


def test_get_pipeline_missing_file_fails(tmp_path):
    result = get_pipeline(str(tmp_path / "missing.yaml"))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# validate_pipeline
# ---------------------------------------------------------------------------

def test_validate_pipeline_valid(tmp_path):
    dep = tmp_path / "raw.csv"
    dep.write_text("a,b\n1,2\n")
    stages = {"prep": {"cmd": "python prep.py", "deps": ["raw.csv"], "outs": []}}
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = validate_pipeline(p, project_root=str(tmp_path))
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_pipeline_missing_cmd(tmp_path):
    stages = {"bad": {"deps": [], "outs": []}}
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = validate_pipeline(p)
    assert result["valid"] is False
    errors = [e["error"] for e in result["errors"]]
    assert "missing cmd" in errors


def test_validate_pipeline_missing_dep_file(tmp_path):
    stages = {"train": {"cmd": "python train.py", "deps": ["ghost.csv"], "outs": []}}
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = validate_pipeline(p, project_root=str(tmp_path))
    assert result["valid"] is False
    assert any(e["error"] == "missing dependency" for e in result["errors"])


def test_validate_pipeline_cycle_detected(tmp_path):
    stages = {
        "a": {"cmd": "cmd_a", "deps": ["b_out"], "outs": ["a_out"]},
        "b": {"cmd": "cmd_b", "deps": ["a_out"], "outs": ["b_out"]},
    }
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = validate_pipeline(p)
    assert result["valid"] is False
    assert any(e["error"] == "circular dependency detected" for e in result["errors"])


def test_validate_pipeline_missing_file_fails(tmp_path):
    result = validate_pipeline(str(tmp_path / "nope.yaml"))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# add_pipeline_stage
# ---------------------------------------------------------------------------

def test_add_pipeline_stage_appends(tmp_path):
    p = write_pipeline(tmp_path / "p.pipeline.yaml", {"existing": MINIMAL_STAGE})
    result = add_pipeline_stage(p, "new_stage", {"cmd": "echo hi"})
    assert result["success"] is True
    loaded = yaml.safe_load(open(p).read())
    assert "new_stage" in loaded["stages"]
    assert "existing" in loaded["stages"]


def test_add_pipeline_stage_duplicate_fails(tmp_path):
    p = write_pipeline(tmp_path / "p.pipeline.yaml", {"dupe": MINIMAL_STAGE})
    result = add_pipeline_stage(p, "dupe", {"cmd": "echo x"})
    assert result["success"] is False
    assert "dupe" in result["error"]


def test_add_pipeline_stage_missing_file_fails(tmp_path):
    result = add_pipeline_stage(str(tmp_path / "ghost.yaml"), "s", {"cmd": "x"})
    assert result["success"] is False


# ---------------------------------------------------------------------------
# remove_pipeline_stage
# ---------------------------------------------------------------------------

def test_remove_pipeline_stage_happy_path(tmp_path):
    p = write_pipeline(tmp_path / "p.pipeline.yaml", {
        "a": MINIMAL_STAGE,
        "b": MINIMAL_STAGE,
    })
    result = remove_pipeline_stage(p, "a")
    assert result["success"] is True
    assert result["removed_stage"] == "a"
    loaded = yaml.safe_load(open(p).read())
    assert "a" not in loaded["stages"]
    assert "b" in loaded["stages"]


def test_remove_pipeline_stage_reports_dependents(tmp_path):
    stages = {
        "prep": {"cmd": "prep", "deps": [], "outs": ["data.csv"]},
        "train": {"cmd": "train", "deps": ["data.csv"], "outs": []},
    }
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = remove_pipeline_stage(p, "prep")
    assert result["success"] is True
    assert "train" in result["dependent_stages"]


def test_remove_pipeline_stage_not_found(tmp_path):
    p = write_pipeline(tmp_path / "p.pipeline.yaml", {"a": MINIMAL_STAGE})
    result = remove_pipeline_stage(p, "ghost")
    assert result["success"] is False
    assert "ghost" in result["error"]


def test_remove_pipeline_stage_missing_file_fails(tmp_path):
    result = remove_pipeline_stage(str(tmp_path / "nope.yaml"), "s")
    assert result["success"] is False


# ---------------------------------------------------------------------------
# get_pipeline_status
# ---------------------------------------------------------------------------

def test_get_pipeline_status_up_to_date(tmp_path):
    dep = tmp_path / "input.csv"
    dep.write_text("a\n1\n")
    out = tmp_path / "output.pkl"
    out.write_bytes(b"model")
    # ensure dep is older than out
    t = time.time()
    dep.stat()  # touch read; set mtime explicitly
    import os
    os.utime(dep, (t - 10, t - 10))
    os.utime(out, (t, t))

    stages = {"train": {"cmd": "train", "deps": ["input.csv"], "outs": ["output.pkl"]}}
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = get_pipeline_status(p, project_root=str(tmp_path))
    assert result["success"] is True
    assert result["stages"][0]["status"] == "up-to-date"


def test_get_pipeline_status_missing_output_is_stale(tmp_path):
    dep = tmp_path / "input.csv"
    dep.write_text("a\n1\n")
    stages = {"train": {"cmd": "train", "deps": ["input.csv"], "outs": ["missing.pkl"]}}
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = get_pipeline_status(p, project_root=str(tmp_path))
    assert result["stages"][0]["status"] == "stale"
    assert result["stages"][0]["reason"] == "missing_outputs"


def test_get_pipeline_status_stale_when_dep_newer(tmp_path):
    import os
    dep = tmp_path / "input.csv"
    dep.write_text("a\n1\n")
    out = tmp_path / "output.pkl"
    out.write_bytes(b"old")
    t = time.time()
    os.utime(out, (t - 20, t - 20))
    os.utime(dep, (t, t))

    stages = {"train": {"cmd": "train", "deps": ["input.csv"], "outs": ["output.pkl"]}}
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = get_pipeline_status(p, project_root=str(tmp_path))
    assert result["stages"][0]["status"] == "stale"
    assert result["stages"][0]["reason"] == "deps_newer_than_outs"


def test_get_pipeline_status_missing_file_fails(tmp_path):
    result = get_pipeline_status(str(tmp_path / "nope.yaml"))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# list_pipelines
# ---------------------------------------------------------------------------

def test_list_pipelines_finds_yaml_files(tmp_path):
    (tmp_path / "a.pipeline.yaml").write_text(yaml.safe_dump({"stages": {}}))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pipeline.yaml").write_text(yaml.safe_dump({"stages": {}}))
    result = list_pipelines(str(tmp_path))
    assert result["success"] is True
    assert result["count"] == 2
    names = result["pipelines"]
    assert any("a.pipeline.yaml" in n for n in names)
    assert any("b.pipeline.yaml" in n for n in names)


def test_list_pipelines_ignores_non_pipeline_yaml(tmp_path):
    (tmp_path / "dvc.yaml").write_text("")
    (tmp_path / "p.pipeline.yaml").write_text(yaml.safe_dump({"stages": {}}))
    result = list_pipelines(str(tmp_path))
    assert result["count"] == 1


def test_list_pipelines_empty_dir(tmp_path):
    result = list_pipelines(str(tmp_path))
    assert result["success"] is True
    assert result["count"] == 0


def test_list_pipelines_bad_path(tmp_path):
    result = list_pipelines(str(tmp_path / "nope"))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# visualize_pipeline
# ---------------------------------------------------------------------------

def test_visualize_pipeline_contains_mermaid_header(tmp_path):
    stages = {
        "prep": {"cmd": "prep", "deps": [], "outs": ["data.csv"]},
        "train": {"cmd": "train", "deps": ["data.csv"], "outs": []},
    }
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = visualize_pipeline(p)
    assert result["success"] is True
    assert result["mermaid"].startswith("graph TD")


def test_visualize_pipeline_edge_in_mermaid(tmp_path):
    stages = {
        "prep": {"cmd": "prep", "deps": [], "outs": ["data.csv"]},
        "train": {"cmd": "train", "deps": ["data.csv"], "outs": []},
    }
    p = write_pipeline(tmp_path / "p.pipeline.yaml", stages)
    result = visualize_pipeline(p)
    assert "prep --> train" in result["mermaid"]


def test_visualize_pipeline_missing_file_fails(tmp_path):
    result = visualize_pipeline(str(tmp_path / "ghost.yaml"))
    assert result["success"] is False


# TODO: test mermaid output for pipelines with multiple fan-out edges
@pytest.mark.skip(reason="multi-fan-out mermaid rendering not yet validated")
def test_visualize_pipeline_fan_out():
    pass
