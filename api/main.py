from fastapi import FastAPI, HTTPException
from api.models import PredictRequest, PredictResponse, Prediction, Event

app = FastAPI(
    title="API Call Predictor",
    description="predicts next best API call",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}
