# src/api/schemas.py

from pydantic import BaseModel
from typing import List


class SymptomItem(BaseModel):
    label: str
    confidence: float


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    risk_label: str
    risk_score: float
    priority: str
    detected_symptoms: List[SymptomItem]
    symptom_severity: str
    psychological_pattern: str
    explanation: str
    disclaimer: str
    emergency_support: dict | None
    system_confidence: str

