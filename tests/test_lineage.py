from pathlib import Path

import pytest

from mlops_mcp.lineage import (
    check_lineage_integrity,
    get_artifact_provenance,
    list_lineage_artifacts,
    record_lineage,
    visualize_lineage,
)


# ── record_lineage ────────────────────────────────────────────────

class TestRecordLineage:
    def test_records_entry_and_persists_to_disk(self, tmp_path):
        r = record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        assert r["success"] is True
        assert (tmp_path / ".mlops" / "lineage.json").exists()

    def test_record_count_increments(self, tmp_path):
        r1 = record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        r2 = record_lineage(str(tmp_path), "report.md", ["model.pt"], "run-2")
        assert r1["record_count"] == 1
        assert r2["record_count"] == 2

    def test_empty_output_returns_error(self, tmp_path):
        r = record_lineage(str(tmp_path), "", ["data.csv"], "run-1")
        assert r["success"] is False

    def test_empty_inputs_returns_error(self, tmp_path):
        r = record_lineage(str(tmp_path), "model.pt", [], "run-1")
        assert r["success"] is False

    def test_empty_run_id_returns_error(self, tmp_path):
        r = record_lineage(str(tmp_path), "model.pt", ["data.csv"], "")
        assert r["success"] is False

    def test_second_record_does_not_overwrite_first(self, tmp_path):
        record_lineage(str(tmp_path), "a.pt", ["x.csv"], "run-1")
        record_lineage(str(tmp_path), "b.pt", ["y.csv"], "run-2")
        r = list_lineage_artifacts(str(tmp_path))
        artifacts = {a["artifact"] for a in r["artifacts"]}
        assert "a.pt" in artifacts
        assert "b.pt" in artifacts


# ── get_artifact_provenance ───────────────────────────────────────

class TestGetArtifactProvenance:
    def test_single_hop_returns_correct_edge(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        r = get_artifact_provenance(str(tmp_path), "model.pt")
        assert r["success"] is True
        assert r["record_count"] == 1
        assert r["provenance"][0]["output"] == "model.pt"
        assert "data.csv" in r["provenance"][0]["inputs"]

    def test_multi_hop_traverses_full_chain(self, tmp_path):
        record_lineage(str(tmp_path), "features.parquet", ["raw.csv"], "run-1")
        record_lineage(str(tmp_path), "model.pt", ["features.parquet"], "run-2")
        r = get_artifact_provenance(str(tmp_path), "model.pt")
        outputs = [e["output"] for e in r["provenance"]]
        assert "model.pt" in outputs
        assert "features.parquet" in outputs

    def test_unknown_artifact_returns_empty_provenance(self, tmp_path):
        r = get_artifact_provenance(str(tmp_path), "ghost.pt")
        assert r["success"] is True
        assert r["record_count"] == 0
        assert r["provenance"] == []
        assert "ghost.pt" in r["source_artifacts"]

    def test_source_artifacts_contains_root_inputs(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        r = get_artifact_provenance(str(tmp_path), "model.pt")
        assert "data.csv" in r["source_artifacts"]


# ── list_lineage_artifacts ────────────────────────────────────────

class TestListLineageArtifacts:
    def test_returns_all_artifacts_outputs_and_inputs(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["train.csv", "config.yaml"], "run-1")
        r = list_lineage_artifacts(str(tmp_path))
        names = {a["artifact"] for a in r["artifacts"]}
        assert "model.pt" in names
        assert "train.csv" in names
        assert "config.yaml" in names

    def test_artifact_type_classified_by_extension(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["data.csv", "params.yaml"], "run-1")
        r = list_lineage_artifacts(str(tmp_path))
        by_name = {a["artifact"]: a["type"] for a in r["artifacts"]}
        assert by_name["model.pt"] == "model"
        assert by_name["data.csv"] == "dataset"
        assert by_name["params.yaml"] == "config"

    def test_empty_lineage_returns_empty_list(self, tmp_path):
        r = list_lineage_artifacts(str(tmp_path))
        assert r["success"] is True
        assert r["artifacts"] == []
        assert r["count"] == 0

    def test_count_matches_artifacts_length(self, tmp_path):
        record_lineage(str(tmp_path), "out.pt", ["a.csv", "b.csv"], "run-1")
        r = list_lineage_artifacts(str(tmp_path))
        assert r["count"] == len(r["artifacts"])


# ── visualize_lineage ─────────────────────────────────────────────

class TestVisualizeLineage:
    def test_mermaid_starts_with_graph_td(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        r = visualize_lineage(str(tmp_path), "model.pt")
        assert r["success"] is True
        assert r["mermaid"].startswith("graph TD")

    def test_edge_appears_as_parent_arrow_output(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        r = visualize_lineage(str(tmp_path), "model.pt")
        assert "data.csv --> model.pt" in r["mermaid"]

    def test_no_lineage_returns_single_node(self, tmp_path):
        r = visualize_lineage(str(tmp_path), "orphan.pt")
        assert r["success"] is True
        assert "orphan.pt[orphan.pt]" in r["mermaid"]
        assert "-->" not in r["mermaid"]

    def test_node_declarations_appear_before_edges(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        r = visualize_lineage(str(tmp_path), "model.pt")
        lines = r["mermaid"].splitlines()
        # node declarations are [name] form, edges contain -->
        node_lines = [i for i, l in enumerate(lines) if "[" in l and "-->" not in l]
        edge_lines = [i for i, l in enumerate(lines) if "-->" in l]
        assert node_lines and edge_lines
        assert max(node_lines) < min(edge_lines)


# ── check_lineage_integrity ───────────────────────────────────────

class TestCheckLineageIntegrity:
    def test_all_inputs_exist_returns_valid(self, tmp_path):
        # create the actual input file so integrity check passes
        (tmp_path / "data.csv").write_text("x\n1\n")
        record_lineage(str(tmp_path), "model.pt", ["data.csv"], "run-1")
        r = check_lineage_integrity(str(tmp_path))
        assert r["success"] is True
        assert r["is_valid"] is True
        assert r["broken_edges"] == []

    def test_missing_input_appears_in_broken_edges(self, tmp_path):
        record_lineage(str(tmp_path), "model.pt", ["missing.csv"], "run-1")
        r = check_lineage_integrity(str(tmp_path))
        assert r["is_valid"] is False
        assert any(e["missing_input"] == "missing.csv" for e in r["broken_edges"])

    def test_broken_edge_count_matches_list(self, tmp_path):
        record_lineage(str(tmp_path), "out.pt", ["a.csv", "b.csv"], "run-1")
        r = check_lineage_integrity(str(tmp_path))
        assert r["broken_edge_count"] == len(r["broken_edges"])

    def test_empty_lineage_returns_valid(self, tmp_path):
        r = check_lineage_integrity(str(tmp_path))
        assert r["success"] is True
        assert r["is_valid"] is True
        assert r["broken_edge_count"] == 0
