from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ._types import err as _fail
from .analysis import classify_file


def _load_table(dataset_path: Path):
    try:
        import pandas as pd
    except ImportError:
        return None, _fail("pandas not installed")

    ext = dataset_path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(dataset_path), None
        if ext == ".tsv":
            return pd.read_csv(dataset_path, sep="\t"), None
        if ext == ".json":
            return pd.read_json(dataset_path), None
        if ext == ".jsonl":
            return pd.read_json(dataset_path, lines=True), None
        if ext == ".parquet":
            return pd.read_parquet(dataset_path), None
        if ext == ".feather":
            return pd.read_feather(dataset_path), None
    except Exception as exc:  # noqa: BLE001
        return None, _fail(str(exc))

    return None, _fail(f"can't read dataset type: {ext}")


def _write_table(frame, output_path: Path) -> dict[str, Any]:
    ext = output_path.suffix.lower()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if ext == ".csv":
            frame.to_csv(output_path, index=False)
            return {"success": True}
        if ext == ".parquet":
            frame.to_parquet(output_path, index=False)
            return {"success": True}
        return _fail("output_path must end with .csv or .parquet")
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def profile_dataset(path: str) -> dict[str, Any]:
    dataset = Path(path).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        return _fail(f"dataset file not found: {dataset}")

    frame, error = _load_table(dataset)
    if error:
        return error

    numeric = frame.select_dtypes(include=["number"])
    numeric_stats = (
        numeric.describe().to_dict() if not numeric.empty else {}
    )
    return {
        "success": True,
        "path": str(dataset),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "null_counts": {col: int(value) for col, value in frame.isna().sum().items()},
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "numeric_stats": numeric_stats,
    }


def validate_dataset_schema(dataset_path: str, schema_path: str) -> dict[str, Any]:
    dataset = Path(dataset_path).expanduser().resolve()
    schema_file = Path(schema_path).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        return _fail(f"dataset file not found: {dataset}")
    if not schema_file.exists() or not schema_file.is_file():
        return _fail(f"schema file missing: {schema_file}")

    frame, error = _load_table(dataset)
    if error:
        return error

    try:
        if schema_file.suffix.lower() in {".yaml", ".yml"}:
            schema = yaml.safe_load(schema_file.read_text(encoding="utf-8"))
        else:
            schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return _fail(str(exc))

    if not isinstance(schema, dict):
        return _fail("schema must be a mapping")
    expected = schema.get("columns", {})

    schema_errors: list[dict[str, str]] = []
    for name, expected_dtype in expected.items():
        if name not in frame.columns:
            schema_errors.append({"column": name, "error": "missing"})
            continue
        actual_dtype = str(frame[name].dtype)
        if str(expected_dtype) != actual_dtype:
            schema_errors.append(
                {
                    "column": name,
                    "error": "dtype_mismatch",
                    "expected": str(expected_dtype),
                    "actual": actual_dtype,
                }
            )

    return {
        "success": True,
        "dataset_path": str(dataset),
        "schema_path": str(schema_file),
        "valid": len(schema_errors) == 0,
        "errors": schema_errors,
    }


def detect_data_drift(reference_path: str, candidate_path: str) -> dict[str, Any]:
    ref = Path(reference_path).expanduser().resolve()
    cand = Path(candidate_path).expanduser().resolve()
    if not ref.exists() or not ref.is_file():
        return _fail(f"reference dataset missing: {ref}")
    if not cand.exists() or not cand.is_file():
        return _fail(f"candidate dataset missing: {cand}")

    ref_df, err = _load_table(ref)
    if err:
        return err
    cand_df, err = _load_table(cand)
    if err:
        return err

    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return _fail("scipy not installed.")

    report: dict[str, Any] = {}
    common = [c for c in ref_df.columns if c in cand_df.columns]
    for column in common:
        ref_col = ref_df[column]
        cand_col = cand_df[column]
        if str(ref_col.dtype).startswith(("int", "float")) and str(cand_col.dtype).startswith(("int", "float")):
            ref_vals = ref_col.dropna().to_numpy()
            cand_vals = cand_col.dropna().to_numpy()
            if len(ref_vals) == 0 or len(cand_vals) == 0:
                report[column] = {"type": "numeric",
                                  "drift": None, "reason": "empty_series"}
                continue
            stat, p_value = ks_2samp(ref_vals, cand_vals)
            report[column] = {
                "type": "numeric",
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < 0.05),
            }
        else:
            ref_freq = ref_col.astype(str).value_counts(normalize=True)
            cand_freq = cand_col.astype(str).value_counts(normalize=True)
            categories = sorted(set(ref_freq.index) | set(cand_freq.index))
            diff = sum(abs(float(ref_freq.get(cat, 0.0)) - float(cand_freq.get(cat, 0.0))) for cat in categories)
            report[column] = {
                "type": "categorical",
                "frequency_distance": diff,
                "drift_detected": bool(diff > 0.2),
            }

    return {
        "success": True,
        "reference_path": str(ref),
        "candidate_path": str(cand),
        "columns_evaluated": len(report),
        "drift_report": report,
    }


def split_dataset(
    dataset_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_column: str | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        return _fail("train_ratio + val_ratio + test_ratio must equal 1.0")

    dataset = Path(dataset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        return _fail(f"dataset file not found: {dataset}")

    frame, error = _load_table(dataset)
    if error:
        return error

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        return _fail("scikit-learn not installed.")

    stratify_values = None
    if stratify_column:
        if stratify_column not in frame.columns:
            return _fail(f"stratify column not found: {stratify_column}")
        stratify_values = frame[stratify_column]

    try:
        train_df, temp_df = train_test_split(
            frame,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=stratify_values,
        )

        adjusted_test = test_ratio / (val_ratio + test_ratio)
        temp_stratify = temp_df[stratify_column] if stratify_column else None
        val_df, test_df = train_test_split(
            temp_df,
            test_size=adjusted_test,
            random_state=random_seed,
            stratify=temp_stratify,
        )

        ext = dataset.suffix.lower()
        if ext not in {".csv", ".parquet"}:
            return _fail("split_dataset currently supports .csv and .parquet output")

        train_path = output / f"train{ext}"
        val_path = output / f"val{ext}"
        test_path = output / f"test{ext}"

        for df, path in zip([train_df, val_df, test_df], [train_path, val_path, test_path]):
            written = _write_table(df, path)
            if not written.get("success"):
                return written

        return {
            "success": True,
            "output_dir": str(output),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "counts": {
                "train": int(train_df.shape[0]),
                "val": int(val_df.shape[0]),
                "test": int(test_df.shape[0]),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))


def merge_datasets(dataset_paths: list[str], output_path: str, deduplicate: bool = False) -> dict[str, Any]:
    if not dataset_paths:
        return _fail("need at least one dataset path")

    frames = []
    for ds_path in dataset_paths:
        dataset = Path(ds_path).expanduser().resolve()
        if not dataset.exists() or not dataset.is_file():
            return _fail(f"dataset file not found: {dataset}")
        frame, error = _load_table(dataset)
        if error:
            return error
        frames.append(frame)

    import pandas as pd

    merged = pd.concat(frames, ignore_index=True)
    if deduplicate:
        merged = merged.drop_duplicates()

    destination = Path(output_path).expanduser().resolve()
    written = _write_table(merged, destination)
    if not written.get("success"):
        return written

    return {
        "success": True,
        "output_path": str(destination),
        "rows": int(merged.shape[0]),
        "columns": int(merged.shape[1]),
        "deduplicate": deduplicate,
    }


def check_data_freshness(path: str, stale_after_hours: int = 24) -> dict[str, Any]:
    dataset = Path(path).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        return _fail(f"dataset file not found: {dataset}")
    if stale_after_hours < 0:
        return _fail("stale_after_hours must be >= 0")

    modified = datetime.fromtimestamp(dataset.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    age_seconds = max(0.0, (now - modified).total_seconds())
    stale = age_seconds > stale_after_hours * 3600
    return {
        "success": True,
        "path": str(dataset),
        "last_modified": modified.isoformat(),
        "age_seconds": age_seconds,
        "stale_after_hours": stale_after_hours,
        "is_stale": stale,
    }


def generate_dataset_card(
    dataset_path: str,
    output_path: str | None = None,
    sample_rows: int = 5,
) -> dict[str, Any]:
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.exists() or not dataset.is_file():
        return _fail(f"dataset file not found: {dataset}")

    profile = profile_dataset(str(dataset))
    if not profile.get("success"):
        return profile

    frame, error = _load_table(dataset)
    if error:
        return error

    output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else dataset.parent / "DATASET_CARD.md"
    )

    payload = {
        "dataset_name": dataset.name,
        "dataset_path": str(dataset),
        "rows": profile["rows"],
        "columns": profile["columns"],
        "dtypes": profile["dtypes"],
        "null_counts": profile["null_counts"],
        "sample": frame.head(sample_rows).to_dict(orient="records"),
    }

    try:
        from jinja2 import Template

        template = Template(
            """
# DATASET_CARD

- Name: {{ dataset_name }}
- Path: {{ dataset_path }}
- Rows: {{ rows }}
- Columns: {{ columns }}

## Schema
{% for name, dtype in dtypes.items() %}- {{ name }}: {{ dtype }}
{% endfor %}
## Null Counts
{% for name, count in null_counts.items() %}- {{ name }}: {{ count }}
{% endfor %}
## Sample Rows
{{ sample }}
"""
        )
        content = template.render(**payload)
    except ImportError:
        lines = [
            "# DATASET_CARD",
            "",
            f"- Name: {payload['dataset_name']}",
            f"- Path: {payload['dataset_path']}",
            f"- Rows: {payload['rows']}",
            f"- Columns: {payload['columns']}",
            "",
            "## Schema",
        ]
        for name, dtype in payload["dtypes"].items():
            lines.append(f"- {name}: {dtype}")
        lines.append("")
        lines.append("## Null Counts")
        for name, count in payload["null_counts"].items():
            lines.append(f"- {name}: {count}")
        lines.append("")
        lines.append("## Sample Rows")
        lines.append(str(payload["sample"]))
        content = "\n".join(lines) + "\n"

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "dataset_path": str(dataset),
            "output_path": str(output),
            "rows": profile["rows"],
            "columns": profile["columns"],
        }
    except OSError as exc:
        return _fail(str(exc))


def find_dataset_files(path: str) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _fail(f"can't scan missing directory: {root}")

    found: list[dict[str, Any]] = []
    try:
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            classified = classify_file(str(item))
            if not classified.get("success"):
                continue
            if classified.get("category") != "dataset":
                continue

            stat = item.stat()
            found.append(
                {
                    "path": str(item.relative_to(root)),
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )

        found.sort(key=lambda entry: entry["path"])
        return {
            "success": True,
            "path": str(root),
            "count": len(found),
            "datasets": found,
        }
    except OSError as exc:
        return _fail(str(exc))
