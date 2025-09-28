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


async def handle_cold_start(spec_url: str, events: list, prompt: str, k: int):
    """Handle cold start scenarios with < 3 events"""

    # If no prompt, return safe exploration endpoints
    if not prompt:
        return [
            Prediction(
                endpoint="GET /api",
                params={},
                score=0.4,
                why="Cold start: Safe exploration of API root",
            ),
            Prediction(
                endpoint="GET /health",
                params={},
                score=0.35,
                why="Cold start: Health check endpoint",
            ),
            Prediction(
                endpoint="GET /users",
                params={},
                score=0.3,
                why="Cold start: Common user resource",
            ),
        ][:k]

    # With prompt, use LLM but apply cold-start scoring
    candidates = []
    if llm_client:
        try:
            candidates = await llm_client.generate_candidates(
                spec_url=spec_url,
                events=events,
                prompt=prompt,
                n=k * 2,  # Generate fewer since we have less context
            )
        except Exception as e:
            print(f"Cold start LLM failed: {e}")

    # If LLM failed, generate prompt-based fallbacks
    if not candidates:
        candidates = generate_prompt_fallbacks(prompt, events)

    # Apply cold-start scoring (no ML model)
    scored = []
    for candidate in candidates:
        score = calculate_cold_start_score(candidate, prompt, events)

        scored.append(
            {
                "endpoint": candidate["endpoint"],
                "params": candidate.get("params", {}),
                "score": score,
                "why": f"Cold start: {candidate.get('reasoning', 'Prompt-based suggestion')}",
            }
        )

    # Apply safety guardrails
    safe = safety_guardrails.validate_predictions(scored, prompt)

    # Sort and return top-k
    safe.sort(key=lambda x: x["score"], reverse=True)

    return [
        Prediction(
            endpoint=p["endpoint"],
            params=p.get("params", {}),
            score=p["score"],
            why=p["why"],
        )
        for p in safe[:k]
    ]


def generate_prompt_fallbacks(prompt: str, events: list):
    """Generate fallback candidates based on prompt keywords"""

    prompt_lower = prompt.lower() if prompt else ""
    candidates = []

    # Common prompt-to-endpoint mappings
    if any(word in prompt_lower for word in ["list", "show", "view", "get"]):
        candidates.extend(
            [
                {
                    "endpoint": "GET /users",
                    "params": {},
                    "reasoning": "List/view request",
                },
                {"endpoint": "GET /api", "params": {}, "reasoning": "General listing"},
            ]
        )

    if any(word in prompt_lower for word in ["create", "add", "new"]):
        candidates.extend(
            [
                {
                    "endpoint": "POST /users",
                    "params": {},
                    "reasoning": "Create request",
                },
                {
                    "endpoint": "GET /users",
                    "params": {},
                    "reasoning": "View before create",
                },
            ]
        )

    if any(word in prompt_lower for word in ["update", "edit", "modify"]):
        candidates.extend(
            [
                {
                    "endpoint": "GET /users",
                    "params": {},
                    "reasoning": "List for update",
                },
                {
                    "endpoint": "PUT /users/{id}",
                    "params": {},
                    "reasoning": "Update request",
                },
            ]
        )

    # Resource-specific suggestions
    for resource in ["user", "invoice", "payment", "order", "product"]:
        if resource in prompt_lower:
            candidates.extend(
                [
                    {
                        "endpoint": f"GET /{resource}s",
                        "params": {},
                        "reasoning": f"{resource} listing",
                    },
                    {
                        "endpoint": f"GET /{resource}s/{{id}}",
                        "params": {},
                        "reasoning": f"{resource} details",
                    },
                ]
            )

    # If we have some events, suggest related endpoints
    if events:
        last_endpoint = events[-1]["endpoint"]
        last_resource = last_endpoint.split("/")[1] if "/" in last_endpoint else "api"
        candidates.append(
            {
                "endpoint": f"GET /{last_resource}",
                "params": {},
                "reasoning": "Continue with same resource",
            }
        )

    return candidates[:10]  # Limit fallbacks


def calculate_cold_start_score(candidate: dict, prompt: str, events: list):
    """Calculate score for cold start without ML model"""

    score = 0.3  # Base cold start score

    endpoint = candidate["endpoint"]
    method = endpoint.split()[0] if endpoint else "GET"

    # Boost safe operations
    if method == "GET":
        score += 0.2
    elif method == "POST":
        score += 0.1

    # Penalize dangerous operations heavily in cold start
    if method == "DELETE":
        score *= 0.1
        if prompt and ("delete" in prompt.lower() or "remove" in prompt.lower()):
            score *= 3  # Less penalty if explicitly requested

    # Boost if prompt alignment
    if prompt:
        reasoning = candidate.get("reasoning", "").lower()
        prompt_words = prompt.lower().split()
        reasoning_words = reasoning.split()

        # Simple word overlap scoring
        overlap = len(set(prompt_words) & set(reasoning_words))
        score += min(overlap * 0.1, 0.3)

    # Boost if continuing same resource
    if events:
        last_endpoint = events[-1]["endpoint"]
        last_resource = last_endpoint.split("/")[1] if "/" in last_endpoint else ""
        cand_resource = endpoint.split("/")[1] if "/" in endpoint else ""

        if last_resource == cand_resource:
            score += 0.15

    return min(score, 0.95)  # Cap at 0.95 for cold start


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "llm_available": llm_client is not None,
        "ml_loaded": ml_predictor.model_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Generate API call predictions"""
    start_time = time.time()
    print(f"[START] REQUEST: {request.user_id}")

    try:
        # 1. Sanitize user input
        sanitize_start = time.time()
        clean_prompt = prompt_sanitizer.sanitize_prompt(request.prompt)
        history = [{"endpoint": e.endpoint, "params": e.params} for e in request.events]
        print(f"[TIMING] SANITIZE: {(time.time() - sanitize_start) * 1000:.1f}ms")

        # 2. Cold start handling for users with < 3 events
        if len(request.events) < 3:
            print(f"[COLDSTART] {len(request.events)} events")
            predictions = await handle_cold_start(
                spec_url=request.spec_url,
                events=history,
                prompt=clean_prompt,
                k=request.k,
            )

            elapsed = time.time() - start_time
            print(
                f"[DONE] Cold start {elapsed * 1000:.1f}ms - {len(predictions)} predictions"
            )
            return PredictResponse(predictions=predictions)

        # 3. Generate candidates via LLM
        llm_start = time.time()
        print(f"[LLM] STARTING")
        candidates = []
        if llm_client:
            try:
                candidates = await llm_client.generate_candidates(
                    spec_url=request.spec_url,
                    events=history,
                    prompt=clean_prompt,
                    n=request.k * 3,  # Generate extra for better selection
                )
                print(f"[TIMING] LLM: {(time.time() - llm_start) * 1000:.1f}ms")
            except Exception as e:
                print(f"[ERROR] LLM failed: {e}")

        # 4. Fallback if no LLM candidates
        if not candidates:
            candidates = [
                {"endpoint": "GET /api", "params": {}, "reasoning": "API root"},
                {"endpoint": "GET /health", "params": {}, "reasoning": "Health check"},
            ]

        # 5. Score candidates with ML model
        ml_start = time.time()
        scored = ml_predictor.rank_candidates(history, candidates, clean_prompt)
        print(f"[TIMING] ML: {(time.time() - ml_start) * 1000:.1f}ms")

        # 6. Apply safety guardrails
        safety_start = time.time()
        safe = safety_guardrails.validate_predictions(scored, clean_prompt)
        print(f"[TIMING] SAFETY: {(time.time() - safety_start) * 1000:.1f}ms")

        # 7. Sort by score and return top-k
        safe.sort(key=lambda x: x["score"], reverse=True)
        top = safe[: request.k]

        # 8. Convert to response format
        convert_start = time.time()
        predictions = [
            Prediction(
                endpoint=p["endpoint"],
                params=p.get("params", {}),
                score=p["score"],
                why=p["why"],
            )
            for p in top
        ]
        print(f"[TIMING] CONVERT: {(time.time() - convert_start) * 1000:.1f}ms")

        elapsed = time.time() - start_time
        print(f"[DONE] TOTAL: {elapsed * 1000:.1f}ms - {len(predictions)} predictions")

        return PredictResponse(predictions=predictions)

    except Exception as e:
        print(f"❌ Error: {e}")
        # Emergency fallback
        return PredictResponse(
            predictions=[
                Prediction(
                    endpoint="GET /health",
                    params={},
                    score=0.3,
                    why="Emergency fallback",
                )
            ]
        )
