from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict
import joblib


def save_artifacts(pipeline: Any, metrics: Dict[str, Any], feature_names: list[str], out_dir: str | Path) -> None:
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipeline, out / "model.joblib")
	(out / "feature_names.json").write_text(json.dumps(feature_names))
	(out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_pipeline(model_dir: str | Path) -> Any:
	path = Path(model_dir) / "model.joblib"
	if not path.exists():
		raise FileNotFoundError(f"Model not found at {path}")
	return joblib.load(path)


def load_feature_names(model_dir: str | Path) -> list[str]:
	path = Path(model_dir) / "feature_names.json"
	if not path.exists():
		raise FileNotFoundError(f"feature_names.json not found at {path}")
	return json.loads(path.read_text())


def load_metrics(model_dir: str | Path) -> Dict[str, Any]:
	path = Path(model_dir) / "metrics.json"
	if not path.exists():
		return {}
	return json.loads(path.read_text())

