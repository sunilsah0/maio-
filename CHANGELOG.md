# Changelog

## v0.1.0 (Baseline)
- Model: StandardScaler + LinearRegression
- Data: scikit-learn diabetes dataset (`load_diabetes`) with held-out split (random_state=42)
- Metric: RMSE reported in training artifacts (see `model/metrics_v0.1.0.json`)
- API: `/health`, `/predict`
- Docker: single service exposing 8080
- CI: PR/push pipeline (lint/tests/smoke), Release pipeline (GHCR + Release)

## v0.2.0 (Improvement)
- To be implemented: candidate models (Ridge/RandomForest), possible preprocessing changes
- Metrics: RMSE deltas vs v0.1.0; if thresholded flag added, report precision/recall at threshold

