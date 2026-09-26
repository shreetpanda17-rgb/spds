# SPDS — Enterprise Intelligence Operating System

A production-oriented autonomous analytics prototype. It is intentionally model/LLM agnostic and performs the core analytical workflow locally without an API key.

## What it automates
- ingestion of CSV, Excel, JSON and Parquet
- data quality profiling and cleaning
- descriptive analytics and correlations
- anomaly detection
- parallel model benchmarking
- time-series forecasting with uncertainty bands
- downside/baseline/expansion scenarios
- evidence-backed executive insights and caveats

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Architecture
The deterministic engine is separated from the UI so it can later sit behind an API/service layer. The LLM is an optional orchestration/explanation layer, not the source of truth for calculations.

## Production roadmap
For enterprise deployment add: PostgreSQL/object storage, Polars/DuckDB for large data, a job queue, model registry, experiment tracking, RBAC/SSO, audit logs, secrets management, tenant isolation, data lineage, connectors, external market-data providers, causal inference, optimization, monitoring, and a model-provider abstraction.
