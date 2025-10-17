# Changelog

## v0.1.0 (Baseline)
- Model: StandardScaler + LinearRegression
- Data: scikit-learn diabetes dataset (`load_diabetes`) with held-out split (random_state=42)
- Metric: RMSE reported in training artifacts (see `model/metrics_v0.1.0.json`)
- API: `/health`, `/predict`
- Docker: single service exposing 8080
- CI: PR/push pipeline (lint/tests/smoke), Release pipeline (GHCR + Release)

## v0.2.0 (Improvement)
- Added model selection with `--model` flag: `linear`, `ridge`, `rf`, or `auto` (selects best RMSE)
- Candidate models: Ridge (with scaling), RandomForestRegressor (no scaling), LinearRegression baseline
- Reproducible split with `random_state=42`, `test_size=0.2`
- Metrics (local run):
  - v0.1.0 (LinearRegression): RMSE ≈ 53.85
  - v0.2.0 (auto picked Ridge): RMSE ≈ 53.78
  - Delta: -0.07 RMSE (small but consistent gain with regularization)

Rationale: Ridge regularization slightly reduces variance on held-out data. RandomForest did not outperform linear models at these defaults while increasing inference latency and image size; thus Ridge is preferred for this dataset and constraints.

