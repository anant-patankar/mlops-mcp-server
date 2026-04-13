## MLOps MCP Server

[![PyPI](https://img.shields.io/pypi/v/mlops-mcp-server)](https://pypi.org/project/mlops-mcp-server/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**MCP server for common MLOps workflows**

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that gives AI assistants like Claude direct access to MLOps workflows — experiment tracking, model registry, dataset management, pipeline orchestration, lineage tracing, and more. Wraps DVC, MLflow, and Git rather than replacing them.

---

## Features

- **Experiment tracking** — create runs, log params/metrics/artifacts, compare runs, find best run, export to CSV
- **Model registry** — register, promote, tag, deprecate, and compare model versions; generate model cards
- **Dataset management** — profile datasets, validate schemas, detect statistical drift (KS-test), split, merge, and generate dataset cards
- **Pipeline management** — create, validate, and visualize DAG pipelines as Mermaid diagrams; cycle detection via Kahn's algorithm
- **Data lineage** — record artifact lineage, trace provenance with BFS, visualize as Mermaid graphs, check integrity
- **File operations** — full file/directory CRUD, content search, disk usage, batch operations
- **Project scaffolding** — 7 ML project templates with component injection and structure validation
- **Documentation generation** — model cards, dataset cards, experiment reports, pipeline docs, project READMEs, API docs
- **MLflow integration** — optional; wraps tracking, registry, and artifact operations
- **DVC integration** — optional; wraps data versioning and pipeline reproduction
- **Git operations** — optional; status, add, commit, log, .gitignore generation
- **Environment tools** — scan imports, generate requirements, check conflicts, create conda env files

Tools are registered in two tiers: a small set of always-on tools for quick file ops and discovery, and 15 domain modules you activate on demand to avoid flooding the agent context window.

---

## Installation

### Install directly from GitHub (available now)

```bash
pip install git+https://github.com/anant-patankar/mlops-mcp-server.git
```

Using `uv`:
```bash
uv add git+https://github.com/anant-patankar/mlops-mcp-server.git
```

### Install from PyPI (coming soon)

```bash
# Core install
pip install mlops-mcp-server

# With optional extras
pip install mlops-mcp-server[mlflow]
pip install mlops-mcp-server[mlflow,compare,notebooks,parquet]
```

Using `uv`:

```bash
uv add mlops-mcp-server
uv add "mlops-mcp-server[mlflow,compare,notebooks,parquet]"
```

---

## Usage

### Claude Desktop

Add to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

First install the package, then add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "mlops": {
      "command": "mlops-mcp-server"
    }
  }
}
```

Restart Claude Desktop. The server starts automatically when Claude connects.

### Example prompts

Once connected, you can ask Claude things like:

> "Show me all experiment runs, find the one with the best validation accuracy, and register that model in the registry."

> "Profile `data/train.csv`, check it against `schema.yaml`, then split it 70/15/15 and save to `data/splits/`."

> "Create a pipeline with three stages: preprocess → train → evaluate, validate it for cycles, and show me the Mermaid diagram."

### Running directly

```bash
mlops-mcp-server
```

---

## Optional Dependencies

| Extra | Installs | Enables |
|-------|----------|---------|
| `mlflow` | mlflow | MLflow tracking and registry tools |
| `compare` | deepdiff | Structured diff in `compare_runs` |
| `notebooks` | nbformat | Notebook summary in `get_notebook_summary` |
| `parquet` | pyarrow | Parquet read/write in dataset tools |
| `models` | joblib, onnx, safetensors | Model file format support |
| `validation` | pandera, scipy | Schema validation and drift statistics |
| `templates` | jinja2 | Jinja2 templates in `create_model_card` |

All extras are optional — the server runs without them and returns clear errors if a missing dependency is needed.

---

## License

MIT — see [LICENSE](LICENSE).
