from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


class Event(BaseModel):
    ts: datetime
    endpoint: str
    params: Dict = {}


class PredictRequest(BaseModel):
    user_id: str
    events: List[Event]
    prompt: Optional[str] = None
    spec_url: str
    k: int = 5


class Prediction(BaseModel):
    endpoint: str
    params: Dict
    score: float
    why: str


class PredictResponse(BaseModel):
    predictions: List[Prediction]

