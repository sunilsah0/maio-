from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .model_io import save_artifacts


@dataclass(frozen=True)
class TrainConfig:
	version: str
	random_state: int
	test_size: float
	out_dir: Path


def build_pipeline() -> Pipeline:
	return Pipeline(
		steps=[
			("scaler", StandardScaler()),
			("model", LinearRegression(n_jobs=None)),
		]
	)


def train_and_evaluate(cfg: TrainConfig) -> Tuple[Pipeline, Dict[str, Any], list[str]]:
	# Reproducibility
	rng = np.random.RandomState(cfg.random_state)
	np.random.seed(cfg.random_state)

	Xy = load_diabetes(as_frame=True)
	X = Xy.frame.drop(columns=["target"])  # 10 features
	y = Xy.frame["target"]
	feature_names = list(X.columns)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=cfg.test_size, random_state=cfg.random_state
	)

	pipeline = build_pipeline()
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)
	rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

	metrics = {
		"version": cfg.version,
		"random_state": cfg.random_state,
		"test_size": cfg.test_size,
		"rmse": rmse,
	}
	return pipeline, metrics, feature_names


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--version", type=str, required=True)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--out-dir", type=str, default="model")
	args = parser.parse_args()

	cfg = TrainConfig(
		version=args.version,
		random_state=args.random_state,
		test_size=args.test_size,
		out_dir=Path(args.out_dir),
	)

	pipeline, metrics, feature_names = train_and_evaluate(cfg)
	save_artifacts(pipeline, metrics, feature_names, cfg.out_dir)

	print({"rmse": metrics["rmse"], "out_dir": str(cfg.out_dir)})


if __name__ == "__main__":
	main()

