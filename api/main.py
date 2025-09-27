import time
from fastapi import FastAPI, HTTPException
from api.models import PredictRequest, PredictResponse, Prediction
from ml.data_generator import TrainingDataGenerator


app = FastAPI(
    title="API Call Predictor",
    description="predicts next best API call",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Health check"""
    generator = TrainingDataGenerator()
    print(generator.generate_training_samples())
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: PredictRequest):
    start_time = time.time()
    print(request)

    try:
        mock_predictions = [
            Prediction(
                endpoint="GET /api/placeholder",
                params={},
                score=0.5,
                why="Mock prediction",
            )
        ]

        elapsed = time.time() - start_time
        print(f"Request processed in {elapsed:.2f}s")

        return PredictResponse(predictions=mock_predictions)

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

