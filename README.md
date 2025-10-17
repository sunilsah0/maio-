# Virtual Diabetes Clinic Triage ML Service

A small FastAPI service that predicts short-term diabetes progression risk using the scikit-learn diabetes dataset. Built with CI/CD via GitHub Actions and packaged as a Docker image.

## Versions
- v0.1: StandardScaler + LinearRegression baseline. Metric: RMSE on held-out split.
- v0.2: Improvements (Ridge/RandomForest and/or preprocessing). Metrics compared in `CHANGELOG.md`.

## How to run (Docker Compose)
```bash
# Pull images published to GHCR (after release)
docker compose up -d
# Or build locally
docker compose -f docker-compose.local.yml up -d --build
```

Service will be on http://localhost:8080

## Health
```bash
curl http://localhost:8080/health
# -> {"status":"ok","model_version":"0.1.0"}
```

## Predict
Assuming inputs are pre-scaled (as per assignment clarification).
```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
    "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
  }'
# -> {"prediction": <float>}
```

Exact field names are in `src/maio/schemas.py` and match `load_diabetes(as_frame=True)`.

## Develop locally
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e .[dev]
pytest
ruff check .
```

## Train models
```bash
# v0.1 (baseline)
python -m maio.train --version 0.1.0 --random-state 42 --test-size 0.2 --out-dir model

# v0.2 (improvements, auto-picks best of linear/ridge/rf)
python -m maio.train --version 0.2.0 --random-state 42 --test-size 0.2 --out-dir model_v02 --model auto
```
Artifacts (pipeline, metrics, feature names) are saved under `model/` or the specified output directory.

## CI/CD
- On PR/push: lint, tests, training smoke, upload artifacts.
- On tag `v*`: build Docker image, run container smoke tests, push to GHCR, publish GitHub Release with metrics and CHANGELOG excerpt.

## License
MIT

