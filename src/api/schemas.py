# src/api/schemas.py

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    risk_label: str
    risk_score: float
    priority: str
    explanation: str  # Add this line