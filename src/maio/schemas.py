from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict


FEATURE_NAMES = [
	"age",
	"sex",
	"bmi",
	"bp",
	"s1",
	"s2",
	"s3",
	"s4",
	"s5",
	"s6",
]


class PredictRequest(BaseModel):
	age: float = Field(...)
	sex: float = Field(...)
	bmi: float = Field(...)
	bp: float = Field(...)
	s1: float = Field(...)
	s2: float = Field(...)
	s3: float = Field(...)
	s4: float = Field(...)
	s5: float = Field(...)
	s6: float = Field(...)

	def to_row(self) -> list[float]:
		return [getattr(self, name) for name in FEATURE_NAMES]


class PredictResponse(BaseModel):
	prediction: float


class HealthResponse(BaseModel):
	status: str
	model_version: str


class ErrorResponse(BaseModel):
	detail: str | Dict[str, str]

