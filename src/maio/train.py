from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model_io import save_artifacts


@dataclass(frozen=True)
class TrainConfig:
	version: str
	random_state: int
	test_size: float
	out_dir: Path
	model: str  # 'linear' | 'ridge' | 'rf' | 'auto'


def build_pipeline(model: str, random_state: int) -> Pipeline:
	if model == "linear":
		estimator = LinearRegression(n_jobs=None)
		scaler_first = True
	elif model == "ridge":
		# Simple ridge with alpha tuned mildly via heuristic
		estimator = Ridge(alpha=1.0, random_state=None)
		scaler_first = True
	elif model == "rf":
		estimator = RandomForestRegressor(
			n_estimators=200,
			random_state=random_state,
			n_jobs=-1,
		)
		scaler_first = False  # tree-based models don't need scaling
	else:
		raise ValueError(f"Unknown model: {model}")

	steps: list[tuple[str, Any]] = []
	if scaler_first:
		steps.append(("scaler", StandardScaler()))
	steps.append(("model", estimator))
	return Pipeline(steps=steps)


def train_and_evaluate(cfg: TrainConfig) -> tuple[Pipeline, dict[str, Any], list[str]]:
	# Reproducibility
	np.random.seed(cfg.random_state)

	Xy = load_diabetes(as_frame=True)
	X = Xy.frame.drop(columns=["target"])  # 10 features
	y = Xy.frame["target"]
	feature_names = list(X.columns)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=cfg.test_size, random_state=cfg.random_state
	)

	model_choices = [cfg.model]
	if cfg.model == "auto":
		model_choices = ["linear", "ridge", "rf"]

	best: dict[str, Any] | None = None
	best_pipeline: Pipeline | None = None

	for model_name in model_choices:
		pipeline = build_pipeline(model_name, cfg.random_state)
		pipeline.fit(X_train, y_train)
		preds = pipeline.predict(X_test)
		rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
		candidate = {"model": model_name, "rmse": rmse}
		if best is None or rmse < best["rmse"]:
			best = candidate
			best_pipeline = pipeline

	assert best is not None and best_pipeline is not None

	metrics = {
		"version": cfg.version,
		"random_state": cfg.random_state,
		"test_size": cfg.test_size,
		"rmse": best["rmse"],
		"model": best["model"],
	}
	return best_pipeline, metrics, feature_names


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--version", type=str, required=True)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--out-dir", type=str, default="model")
	parser.add_argument(
		"--model",
		type=str,
		default="auto",
		choices=["auto", "linear", "ridge", "rf"],
		help="Model choice for v0.2 (auto selects best RMSE)",
	)
	args = parser.parse_args()

	cfg = TrainConfig(
		version=args.version,
		random_state=args.random_state,
		test_size=args.test_size,
		out_dir=Path(args.out_dir),
		model=args.model,
	)

	pipeline, metrics, feature_names = train_and_evaluate(cfg)
	save_artifacts(pipeline, metrics, feature_names, cfg.out_dir)

	print({"rmse": metrics["rmse"], "model": metrics["model"], "out_dir": str(cfg.out_dir)})


if __name__ == "__main__":
	main()

