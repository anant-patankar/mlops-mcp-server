import pytest
from pathlib import Path

from mlops_mcp.analysis import (
    get_file_info,
    classify_file,
    analysis_directory,
    find_duplicate_files,
    storage_report,
    batch_classify,
    get_dataset_stats,
    detect_model_framework,
    compare_files,
    compare_directories,
)


#  get_file_info 

class TestGetFileInfo:
    def test_basic(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("hello world")
        result = get_file_info(str(f))
        assert result["success"] is True
        assert result["size"] == len("hello world")
        assert "md5" in result
        assert "created_at" in result

    def test_missing_file(self):
        r = get_file_info("/nonexistent/path/file.txt")
        assert r["success"] is False

    def test_directory_is_rejected(self, tmp_path):
        r = get_file_info(str(tmp_path))
        assert r["success"] is False

    def test_md5_skipped_for_large_files(self, tmp_path, monkeypatch):
        f = tmp_path / "big.bin"
        f.write_text("x")
        # patch os.stat at the os level so Path.stat() returns a fake large size
        import os
        real_stat = os.stat
        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            # return a stat_result with st_size overridden to 200MB
            return os.stat_result((
                result.st_mode, result.st_ino, result.st_dev,
                result.st_nlink, result.st_uid, result.st_gid,
                200 * 1024 * 1024,  # st_size
                result.st_atime, result.st_mtime, result.st_ctime,
            ))
        monkeypatch.setattr(os, "stat", fake_stat)
        r = get_file_info(str(f))
        assert r.get("md5_skipped") is True
        assert r["md5"] is None


#  classify_file 

class TestClassifyFile:
    @pytest.mark.parametrize("ext,expected", [
        (".csv",         "dataset"),
        (".parquet",     "dataset"),
        (".jsonl",       "dataset"),
        (".pt",          "model"),
        (".safetensors", "model"),
        (".onnx",        "model"),
        (".ipynb",       "notebook"),
        (".yaml",        "config"),
        (".toml",        "config"),
        (".py",          "script"),
        (".md",          "doc"),
        (".png",         "image"),
        (".log",         "log"),
        (".xyz",         "other"),
    ])
    def test_extensions(self, tmp_path, ext, expected):
        f = tmp_path / f"file{ext}"
        f.write_text("data")
        r = classify_file(str(f))
        assert r["success"] is True
        assert r["category"] == expected

    def test_missing(self):
        r = classify_file("/no/such/file.csv")
        assert r["success"] is False

    def test_uppercase_extension_is_normalised(self, tmp_path):
        # real files sometimes have .CSV or .PT
        f = tmp_path / "DATA.CSV"
        f.write_text("a,b,c")
        r = classify_file(str(f))
        assert r["category"] == "dataset"


#  analyze_directory 

class TestAnalyzeDirectory:
    def test_counts(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.csv").write_text("y")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("z")

        r = analysis_directory(str(tmp_path))
        assert r["success"] is True
        assert r["file_count"] == 3
        assert r["directory_count"] == 1

    def test_missing_dir(self):
        r = analysis_directory("/no/such/dir")
        assert r["success"] is False

    def test_empty_dir(self, tmp_path):
        r = analysis_directory(str(tmp_path))
        assert r["file_count"] == 0
        assert r["total_bytes"] == 0


#  find_duplicate_files 

class TestFindDuplicateFiles:
    def test_finds_duplicates(self, tmp_path):
        content = b"same content here"
        (tmp_path / "a.bin").write_bytes(content)
        (tmp_path / "b.bin").write_bytes(content)
        (tmp_path / "c.bin").write_bytes(b"different")

        r = find_duplicate_files(str(tmp_path), min_size=1)
        assert r["success"] is True
        assert r["group_count"] == 1
        group = r["duplicate_groups"][0]
        assert len(group) == 2

    def test_no_duplicates(self, tmp_path):
        (tmp_path / "x.txt").write_text("aaa")
        (tmp_path / "y.txt").write_text("bbb")
        r = find_duplicate_files(str(tmp_path), min_size=1)
        assert r["group_count"] == 0

    def test_min_size_filter_excludes_small_files(self, tmp_path):
        # two identical tiny files, but they're below min_size
        (tmp_path / "a.txt").write_text("hi")
        (tmp_path / "b.txt").write_text("hi")
        r = find_duplicate_files(str(tmp_path), min_size=9999)
        assert r["group_count"] == 0


#  storage_report 

class TestStorageReport:
    def test_top_k_respected(self, tmp_path):
        for i in range(5):
            f = tmp_path / f"file{i}.txt"
            f.write_bytes(b"x" * (i + 1) * 100)

        r = storage_report(str(tmp_path), top_k=3)
        assert r["success"] is True
        assert len(r["largest_files"]) == 3

    def test_duplicate_savings_calculation(self, tmp_path):
        content = b"repeated content" * 100  # 1600 bytes — above min_size threshold
        (tmp_path / "copy1.bin").write_bytes(content)
        (tmp_path / "copy2.bin").write_bytes(content)

        r = storage_report(str(tmp_path), top_k=10)
        assert r["duplicate_savings_bytes"] == len(content)

    def test_missing_dir(self):
        r = storage_report("/no/such/dir")
        assert r["success"] is False


#  batch_classify 

class TestBatchClassify:
    def test_mixed_valid_invalid(self, tmp_path):
        good = tmp_path / "model.pt"
        good.write_text("weights")

        r = batch_classify([str(good), "/ghost/path.csv"])
        assert r["success"] is True
        assert r["count"] == 1
        assert len(r["failures"]) == 1

    def test_all_valid(self, tmp_path):
        files = []
        for name in ["a.csv", "b.pt", "c.md"]:
            f = tmp_path / name
            f.write_text("content")
            files.append(str(f))

        r = batch_classify(files)
        assert r["count"] == 3
        assert r["failures"] == []


#  get_dataset_stats 

class TestGetDatasetStats:
    def test_csv_basic(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        r = get_dataset_stats(str(f))
        assert r["success"] is True
        assert r["rows"] == 2
        assert r["columns"] == 3

    def test_csv_with_pipe_in_column_names(self, tmp_path):
        # this was the real-world bug — NSE data uses | in col names
        f = tmp_path / "nse.csv"
        f.write_text("NSE_EQ|INE585B01010,NSE_EQ|INE467B01029\n100,200\n")
        r = get_dataset_stats(str(f))
        assert r["success"] is True
        assert r["rows"] == 1

    def test_null_counts_present(self, tmp_path):
        f = tmp_path / "nulls.csv"
        f.write_text("x,y\n1,\n2,3\n")
        r = get_dataset_stats(str(f))
        assert r["success"] is True
        assert "null_counts" in r

    def test_missing_file(self):
        r = get_dataset_stats("/no/such/file.csv")
        assert r["success"] is False

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.xls"
        f.write_text("old excel")
        r = get_dataset_stats(str(f))
        assert r["success"] is False


#  detect_model_framework 

class TestDetectModelFramework:
    @pytest.mark.parametrize("ext,framework", [
        (".pt",          "pytorch"),
        (".pth",         "pytorch"),
        (".ckpt",        "pytorch"),
        (".onnx",        "onnx"),
        (".joblib",      "sklearn"),
        (".pkl",         "sklearn"),
        (".safetensors", "safetensors"),
    ])
    def test_known_frameworks(self, tmp_path, ext, framework):
        f = tmp_path / f"model{ext}"
        f.write_text("fake model")
        r = detect_model_framework(str(f))
        assert r["success"] is True
        assert r["framework"] == framework

    def test_unknown_extension(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_text("weights")
        r = detect_model_framework(str(f))
        assert r["framework"] == "unknown"

    def test_missing_file(self):
        r = detect_model_framework("/no/such/model.pt")
        assert r["success"] is False


#  compare_files 

class TestCompareFiles:
    def test_identical_files(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("same\ncontent\n")
        b.write_text("same\ncontent\n")
        r = compare_files(str(a), str(b))
        assert r["success"] is True
        assert r["identical"] is True
        assert r["added"] == 0
        assert r["removed"] == 0

    def test_different_files(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("line1\nline2\n")
        b.write_text("line1\nchanged\nextra\n")
        r = compare_files(str(a), str(b))
        assert r["identical"] is False
        assert r["added"] > 0
        assert r["removed"] > 0

    def test_missing_file(self, tmp_path):
        a = tmp_path / "real.txt"
        a.write_text("x")
        r = compare_files(str(a), "/ghost.txt")
        assert r["success"] is False


#  compare_directories 

class TestCompareDirectories:
    def test_identical_dirs(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir(); b.mkdir()
        (a / "f.txt").write_text("same")
        (b / "f.txt").write_text("same")
        r = compare_directories(str(a), str(b))
        assert r["success"] is True
        assert r["only_in_a"] == []
        assert r["only_in_b"] == []
        assert r["different_content"] == []

    def test_asymmetric_dirs(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir(); b.mkdir()
        (a / "only_a.txt").write_text("x")
        (b / "only_b.txt").write_text("y")
        r = compare_directories(str(a), str(b))
        assert "only_a.txt" in r["only_in_a"]
        assert "only_b.txt" in r["only_in_b"]

    def test_same_name_different_content(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir(); b.mkdir()
        (a / "config.yaml").write_text("lr: 0.001")
        (b / "config.yaml").write_text("lr: 0.01")
        r = compare_directories(str(a), str(b))
        assert "config.yaml" in r["different_content"]