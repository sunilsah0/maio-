from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

from .schemas import PredictRequest, PredictResponse, HealthResponse, FEATURE_NAMES
from .model_io import load_pipeline, load_feature_names, load_metrics
from . import __version__


MODEL_DIR = os.environ.get("MODEL_DIR", "model")
PORT = int(os.environ.get("PORT", "8080"))

app = FastAPI(title="Diabetes Progression Service", version=__version__)


@app.on_event("startup")
async def startup_event() -> None:
	# Validate model availability and feature alignment
	try:
		_ = load_pipeline(MODEL_DIR)
		trained_features = load_feature_names(MODEL_DIR)
		if trained_features != FEATURE_NAMES:
			raise RuntimeError("Feature mismatch between trained model and API schema.")
	except Exception as exc:  # noqa: BLE001
		raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
	metrics = load_metrics(MODEL_DIR)
	version = (metrics.get("version") if metrics else __version__) or __version__
	return HealthResponse(status="ok", model_version=str(version))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
	try:
		pipeline = load_pipeline(MODEL_DIR)
		row = np.array([req.to_row()], dtype=float)
		pred = float(pipeline.predict(row)[0])
		return PredictResponse(prediction=pred)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc


# Uvicorn entrypoint when run as a module in Docker
def create_app() -> FastAPI:  # for testing
	return app

