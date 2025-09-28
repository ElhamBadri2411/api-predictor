import time
from fastapi import FastAPI, HTTPException
from api.models import PredictRequest, PredictResponse, Prediction
from llm.client import OpenRouterClient
from ml.predictor import APIPredictor
from api.validators import SafetyGuardrails, PromptSanitizer


app = FastAPI(
    title="API Call Predictor",
    description="Intelligent API call prediction using LLM + ML ranking",
    version="1.0.0",
)

# Initialize components once at startup
try:
    llm_client = OpenRouterClient()
except:
    llm_client = None
    print("⚠️ OpenRouter unavailable - using fallbacks")

ml_predictor = APIPredictor()
safety_guardrails = SafetyGuardrails()
prompt_sanitizer = PromptSanitizer()


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "llm_available": llm_client is not None,
        "ml_loaded": ml_predictor.model_loaded
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Generate API call predictions"""
    start_time = time.time()

    try:
        # 1. Sanitize user input
        clean_prompt = prompt_sanitizer.sanitize_prompt(request.prompt)
        history = [{"endpoint": e.endpoint, "params": e.params} for e in request.events]

        # 2. Generate candidates via LLM
        candidates = []
        if llm_client:
            try:
                candidates = await llm_client.generate_candidates(
                    spec_url=request.spec_url,
                    events=history,
                    prompt=clean_prompt,
                    n=request.k * 3  # Generate extra for better selection
                )
            except Exception as e:
                print(f"LLM failed: {e}")

        # 3. Fallback if no LLM candidates
        if not candidates:
            candidates = [
                {"endpoint": "GET /api", "params": {}, "reasoning": "API root"},
                {"endpoint": "GET /health", "params": {}, "reasoning": "Health check"},
            ]

        # 4. Score candidates with ML model
        scored = ml_predictor.rank_candidates(history, candidates, clean_prompt)

        # 5. Apply safety guardrails
        safe = safety_guardrails.validate_predictions(scored, clean_prompt)

        # 6. Sort by score and return top-k
        safe.sort(key=lambda x: x['score'], reverse=True)
        top = safe[:request.k]

        # 7. Convert to response format
        predictions = [
            Prediction(
                endpoint=p['endpoint'],
                params=p.get('params', {}),
                score=p['score'],
                why=p['why']
            )
            for p in top
        ]

        elapsed = time.time() - start_time
        print(f"✅ {elapsed:.3f}s - {len(predictions)} predictions")

        return PredictResponse(predictions=predictions)

    except Exception as e:
        print(f"❌ Error: {e}")
        # Emergency fallback
        return PredictResponse(predictions=[
            Prediction(
                endpoint="GET /health",
                params={},
                score=0.3,
                why="Emergency fallback"
            )
        ])
