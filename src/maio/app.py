from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException

from . import __version__
from .model_io import load_feature_names, load_metrics, load_pipeline
from .schemas import FEATURE_NAMES, HealthResponse, PredictRequest, PredictResponse


def get_model_dir() -> str:
	return os.environ.get("MODEL_DIR", "model")

PORT = int(os.environ.get("PORT", "8080"))

app = FastAPI(title="Diabetes Progression Service", version=__version__)


@app.on_event("startup")
async def startup_event() -> None:
	# Validate model availability and feature alignment
	try:
		model_dir = get_model_dir()
		_ = load_pipeline(model_dir)
		trained_features = load_feature_names(model_dir)
		if trained_features != FEATURE_NAMES:
			raise RuntimeError("Feature mismatch between trained model and API schema.")
	except Exception as exc:  # noqa: BLE001
		raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
	model_dir = get_model_dir()
	metrics = load_metrics(model_dir)
	version = (metrics.get("version") if metrics else __version__) or __version__
	return HealthResponse(status="ok", model_version=str(version))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
	try:
		model_dir = get_model_dir()
		pipeline = load_pipeline(model_dir)
		row = np.array([req.to_row()], dtype=float)
		pred = float(pipeline.predict(row)[0])
		return PredictResponse(prediction=pred)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=str(exc)) from exc


# Uvicorn entrypoint when run as a module in Docker
def create_app() -> FastAPI:  # for testing
	return app

