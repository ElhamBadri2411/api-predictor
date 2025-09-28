"""
ML Model Predictor for Runtime Inference

Loads the trained Logistic Regression model and feature extractor to score
API candidate predictions in real-time. Optimized for speed (< 1ms inference).
"""

import joblib
import numpy as np
import os
from typing import List, Dict, Optional
from ml.features import FeatureExtractor


class APIPredictor:
    """ML predictor for ranking API candidates"""

    def __init__(self, model_path: str = 'data/model.pkl',
                 extractor_path: str = 'data/feature_extractor.pkl'):
        """Initialize predictor with trained model"""

        self.model = None
        self.extractor = None
        self.model_loaded = False

        # Try to load trained model
        try:
            if os.path.exists(model_path) and os.path.exists(extractor_path):
                self.model = joblib.load(model_path)
                self.extractor = joblib.load(extractor_path)
                self.model_loaded = True
                print(f"✅ Loaded trained model from {model_path}")
            else:
                print(f"⚠️ No trained model found at {model_path}")
                print("   Using fallback heuristic scoring")
                self.extractor = FeatureExtractor()

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Using fallback heuristic scoring")
            self.extractor = FeatureExtractor()

    def rank_candidates(self, history: List[Dict], candidates: List[Dict],
                        prompt: Optional[str] = None) -> List[Dict]:
        """Score and rank API candidates

        Args:
            history: List of previous API calls [{"endpoint": "GET /users"}, ...]
            candidates: List of candidate API calls from LLM
            prompt: Optional user prompt for context

        Returns:
            List of scored candidates: [{"endpoint": "...", "score": 0.85, "why": "..."}, ...]
        """

        if self.model_loaded:
            return self._ml_ranking(history, candidates, prompt)
        else:
            return self._heuristic_ranking(history, candidates, prompt)

    def _ml_ranking(self, history: List[Dict], candidates: List[Dict],
                    prompt: Optional[str]) -> List[Dict]:
        """Use trained ML model for ranking"""

        scored_candidates = []

        for candidate in candidates:
            # Extract features
            features = self.extractor.extract(history, candidate, prompt)

            # Get probability score from model
            try:
                # predict_proba returns [prob_class_0, prob_class_1]
                prob_positive = self.model.predict_proba([features])[0][1]
                score = float(prob_positive)

                # Generate explanation based on score
                explanation = self._generate_ml_explanation(score, features, candidate)

            except Exception as e:
                print(f"Warning: ML prediction failed for {candidate.get('endpoint', 'unknown')}: {e}")
                score = 0.5  # Neutral fallback
                explanation = "ML prediction failed, using neutral score"

            scored_candidates.append({
                'endpoint': candidate.get('endpoint', ''),
                'params': candidate.get('params', {}),
                'score': score,
                'why': explanation
            })

        return scored_candidates

    def _heuristic_ranking(self, history: List[Dict], candidates: List[Dict],
                           prompt: Optional[str]) -> List[Dict]:
        """Fallback heuristic when no trained model available"""

        scored_candidates = []

        for candidate in candidates:
            score = 0.5  # Base score
            reasons = []

            # Same resource bonus
            if history:
                last_resource = self._extract_resource(history[-1].get('endpoint', ''))
                cand_resource = self._extract_resource(candidate.get('endpoint', ''))
                if last_resource and last_resource == cand_resource:
                    score += 0.15
                    reasons.append("same resource")

            # Prompt alignment bonus
            if prompt and candidate.get('reasoning'):
                prompt_words = set(prompt.lower().split())
                reasoning_words = set(candidate.get('reasoning', '').lower().split())
                if prompt_words & reasoning_words:  # Intersection
                    score += 0.15
                    reasons.append("prompt match")

            # HTTP method patterns
            endpoint = candidate.get('endpoint', '')
            if 'GET' in endpoint:
                score += 0.1
                reasons.append("safe read")
            elif 'POST' in endpoint:
                score += 0.05
                reasons.append("create action")
            elif 'DELETE' in endpoint:
                score *= 0.3  # Penalize destructive
                reasons.append("destructive (penalized)")

            # Resource depth pattern
            if '/{id}' in endpoint or any(c.isdigit() for c in endpoint):
                score += 0.1
                reasons.append("specific resource")

            explanation = f"Heuristic: {', '.join(reasons) if reasons else 'baseline'}"

            scored_candidates.append({
                'endpoint': candidate.get('endpoint', ''),
                'params': candidate.get('params', {}),
                'score': min(score, 0.95),  # Cap at 95%
                'why': explanation
            })

        return scored_candidates

    def _generate_ml_explanation(self, score: float, features: np.ndarray,
                                 candidate: Dict) -> str:
        """Generate human-readable explanation for ML prediction"""

        endpoint = candidate.get('endpoint', '')

        if score > 0.8:
            if 'GET' in endpoint:
                return "High confidence: Strong navigation pattern match"
            elif 'POST' in endpoint:
                return "High confidence: Likely next action after current sequence"
            elif 'PUT' in endpoint:
                return "High confidence: Update pattern following detail view"
            else:
                return "High confidence: Strong historical pattern match"

        elif score > 0.6:
            if '/{id}' in endpoint:
                return "Medium confidence: Resource-specific action fits context"
            else:
                return "Medium confidence: Partial pattern alignment"

        elif score > 0.4:
            return "Low confidence: Weak pattern match or unusual sequence"

        else:
            if 'DELETE' in endpoint:
                return "Very low confidence: Destructive action without clear intent"
            else:
                return "Very low confidence: Pattern doesn't match user behavior"

    def _extract_resource(self, endpoint: str) -> str:
        """Extract resource name from endpoint"""
        if not endpoint:
            return ""

        # Handle "METHOD /resource/..." format
        parts = endpoint.split('/')
        if len(parts) >= 2:
            return parts[1]  # First part after /
        return ""

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""

        info = {
            'model_loaded': self.model_loaded,
            'model_type': 'Logistic Regression' if self.model_loaded else 'Heuristic',
            'feature_count': self.extractor.n_features if self.extractor else 0
        }

        if self.model_loaded and hasattr(self.model, 'coef_'):
            info['model_coefficients'] = len(self.model.coef_[0])
            info['model_classes'] = self.model.classes_.tolist()

        return info


def load_predictor() -> APIPredictor:
    """Factory function to create and load predictor"""
    return APIPredictor()


# Quick test function
def test_predictor():
    """Test the predictor with sample data"""

    predictor = APIPredictor()

    # Sample data
    history = [
        {"endpoint": "GET /users"},
        {"endpoint": "GET /users/123"}
    ]

    candidates = [
        {"endpoint": "PUT /users/123", "params": {}, "reasoning": "update user"},
        {"endpoint": "GET /users/123/posts", "params": {}, "reasoning": "view user posts"},
        {"endpoint": "DELETE /users/123", "params": {}, "reasoning": "remove user"}
    ]

    prompt = "update user information"

    # Get predictions
    results = predictor.rank_candidates(history, candidates, prompt)

    print("🧪 Predictor Test Results:")
    print(f"Model info: {predictor.get_model_info()}")
    print("\nRanked predictions:")

    for i, result in enumerate(sorted(results, key=lambda x: x['score'], reverse=True), 1):
        print(f"{i}. {result['endpoint']} (score: {result['score']:.3f})")
        print(f"   Why: {result['why']}")

    return results


if __name__ == "__main__":
    test_predictor()