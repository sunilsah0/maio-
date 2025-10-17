from __future__ import annotations

import os
from pathlib import Path
from fastapi.testclient import TestClient

from maio.app import create_app
from maio.train import TrainConfig, train_and_evaluate
from maio.model_io import save_artifacts


def _ensure_test_model(tmpdir: Path) -> None:
	pipeline, metrics, feature_names = train_and_evaluate(
		TrainConfig(version="0.1.0", random_state=42, test_size=0.2, out_dir=tmpdir)
	)
	save_artifacts(pipeline, metrics, feature_names, tmpdir)


def test_health(tmp_path: Path) -> None:
	_ensure_test_model(tmp_path)
	os.environ["MODEL_DIR"] = str(tmp_path)
	app = create_app()
	client = TestClient(app)
	resp = client.get("/health")
	assert resp.status_code == 200
	data = resp.json()
	assert data["status"] == "ok"
	assert "model_version" in data


def test_predict(tmp_path: Path) -> None:
	_ensure_test_model(tmp_path)
	os.environ["MODEL_DIR"] = str(tmp_path)
	app = create_app()
	client = TestClient(app)
	payload = {
		"age": 0.02,
		"sex": -0.044,
		"bmi": 0.06,
		"bp": -0.03,
		"s1": -0.02,
		"s2": 0.03,
		"s3": -0.02,
		"s4": 0.02,
		"s5": 0.02,
		"s6": -0.001,
	}
	resp = client.post("/predict", json=payload)
	assert resp.status_code == 200
	data = resp.json()
	assert "prediction" in data
	assert isinstance(data["prediction"], (float, int))

