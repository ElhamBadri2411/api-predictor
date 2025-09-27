"""
API Prediction Feature Extractor

This module extracts 51-dimensional feature vectors from API usage history and candidate endpoints
to train an ML model for predicting the next API call a user is likely to make.

FEATURE BREAKDOWN (51 total features):

1. SEQUENCE FEATURES (10 features) - Features 0-9
   - Features 0-4: Method distribution in recent history (GET, POST, PUT, PATCH, DELETE counts/percentages)
   - Feature 5: Resource stickiness (1.0 if user stays on same resource, 0.0 otherwise)
   - Feature 6: Depth progression (1.0 if going deeper in URL hierarchy, 0.0 otherwise)
   - Feature 7: Loop detection (1.0 if repeating same endpoint, 0.0 otherwise)
   - Feature 8: Read vs write ratio (GET calls / total calls)
   - Feature 9: Velocity placeholder (currently always 1.0, reserved for future timing features)

2. TRANSITION FEATURES (25 features) - Features 10-34
   - 5x5 transition matrix encoding all possible HTTP method transitions
   - One-hot encoded: last_method → candidate_method
   - Examples: GET→GET, GET→POST, POST→GET, PUT→DELETE, etc.
   - Captures learned patterns like "after POST, often GET to view created resource"

3. RESOURCE FEATURES (5 features) - Features 35-39
   - Feature 35: Same resource (1.0 if candidate uses same resource as last call)
   - Feature 36: Resource similarity (1.0 exact, 0.3 partial, 0.0 different)
   - Feature 37: Depth change (normalized difference in URL path depth)
   - Feature 38: Has ID in URL (1.0 if candidate contains /{id} or numeric ID)
   - Feature 39: Resource in recent history (1.0 if resource appeared recently)

4. PROMPT FEATURES (6 features) - Features 40-45
   - Feature 40: Has prompt (1.0 if user provided natural language prompt)
   - Feature 41: Verb alignment (how well prompt verbs match HTTP method)
   - Feature 42: Resource mentioned (1.0 if prompt mentions candidate resource)
   - Feature 43: Prompt complexity (word count / 20, normalized)
   - Feature 44: Has action words (1.0 if prompt contains create/update/delete/etc.)
   - Feature 45: Safety check (1.0 if destructive operation without permission)

5. PATTERN FEATURES (5 features) - Features 46-50
   - Feature 46: List to detail pattern (GET /resource → GET /resource/{id})
   - Feature 47: Detail to update pattern (GET /resource/{id} → PUT /resource/{id})
   - Feature 48: Create to view pattern (POST /resource → GET /resource/{id})
   - Feature 49: Update to view pattern (PUT /resource/{id} → GET /resource/{id})
   - Feature 50: Safe operation (1.0 for GET/POST, 0.0 for PUT/DELETE)

DESIGN RATIONALE:
- Features capture both syntactic patterns (URL structure, HTTP methods) and semantic patterns (user intent from prompts)
- Balanced representation across different aspects: sequence, transitions, resources, prompts, patterns
- Binary and normalized continuous features for stable ML training
- Cold start handling: returns zero vector when no history available
- Endpoint normalization: converts IDs to {id} placeholder for pattern matching

EXAMPLE FEATURE EXTRACTION:
History: ['GET /users', 'GET /users/123']
Candidate: 'PUT /users/123'
Prompt: 'update user info'
→ High scores for: same_resource, detail_to_update_pattern, verb_alignment, has_action_words
"""

import re
from typing import List, Dict, Optional
import numpy as np


class FeatureExtractor:
    """Extract features for ML model from API history and candidates"""

    def __init__(self):
        self.methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        self.n_features = 51

    def extract(
        self, history: List[Dict], candidate: Dict, prompt: Optional[str]
    ) -> np.ndarray:
        """Extract feature vector from context"""

        # Normalize endpoints
        if history:
            history = [
                {"endpoint": self.normalize_endpoint(h["endpoint"])} for h in history
            ]
        candidate = {"endpoint": self.normalize_endpoint(candidate["endpoint"])}

        features = []

        # Handle cold start
        if not history:
            return np.zeros(self.n_features)

        # 1. Sequence features (10)
        features.extend(self._sequence_features(history))

        # 2. Transition features (25)
        features.extend(self._transition_features(history[-1], candidate))

        # 3. Resource features (5)
        features.extend(self._resource_features(history, candidate))

        # 4. Prompt features (6)
        features.extend(self._prompt_features(candidate, prompt))

        # 5. Pattern features (5)
        features.extend(self._pattern_features(history, candidate))

        return np.array(features)

    def normalize_endpoint(self, endpoint: str) -> str:
        """Normalize IDs in endpoints"""
        # UUIDs
        endpoint = re.sub(r"/[a-f0-9-]{36}", "/{id}", endpoint, flags=re.I)
        # Numeric IDs
        endpoint = re.sub(r"/\d+", "/{id}", endpoint)
        # Alphanumeric IDs
        endpoint = re.sub(r"/[a-z]+_[A-Za-z0-9]{6,}", "/{id}", endpoint)
        return endpoint

    def _sequence_features(self, history: List[Dict]) -> List[float]:
        """Features from recent history"""
        features = []
        recent = history[-3:] if len(history) >= 3 else history

        # Method distribution
        for method in self.methods:
            count = sum(1 for h in recent if method in h["endpoint"])
            features.append(count / len(recent) if recent else 0)

        # Resource stickiness
        if len(history) >= 2:
            resources = [self._get_resource(h["endpoint"]) for h in history[-3:]]
            same_resource = len(set(resources)) == 1
            features.append(1.0 if same_resource else 0.0)
        else:
            features.append(0.0)

        # Depth progression
        if len(history) >= 2:
            depths = [h["endpoint"].count("/") for h in history[-2:]]
            going_deeper = depths[-1] > depths[0]
            features.append(1.0 if going_deeper else 0.0)
        else:
            features.append(0.0)

        # Is looping?
        if len(history) >= 2:
            is_loop = history[-1]["endpoint"] == history[-2]["endpoint"]
            features.append(1.0 if is_loop else 0.0)
        else:
            features.append(0.0)

        # Read vs write ratio
        all_methods = [self._get_method(h["endpoint"]) for h in history]
        read_ratio = all_methods.count("GET") / len(all_methods) if all_methods else 0
        features.append(read_ratio)

        # Velocity placeholder
        features.append(1.0)

        return features

    def _transition_features(self, last: Dict, candidate: Dict) -> List[float]:
        """Method transition matrix (5x5 = 25 features)"""
        features = []

        last_method = self._get_method(last["endpoint"])
        cand_method = self._get_method(candidate["endpoint"])

        # One-hot encode all transitions
        for from_m in self.methods:
            for to_m in self.methods:
                is_transition = last_method == from_m and cand_method == to_m
                features.append(1.0 if is_transition else 0.0)

        return features

    def _resource_features(self, history: List[Dict], candidate: Dict) -> List[float]:
        """Resource relationship features"""
        features = []

        last = history[-1]
        last_res = self._get_resource(last["endpoint"])
        cand_res = self._get_resource(candidate["endpoint"])

        # Same resource?
        features.append(1.0 if last_res == cand_res else 0.0)

        # Similar resource?
        similarity = (
            1.0
            if last_res == cand_res
            else 0.3
            if last_res in cand_res or cand_res in last_res
            else 0.0
        )
        features.append(similarity)

        # Depth change
        depth_change = candidate["endpoint"].count("/") - last["endpoint"].count("/")
        features.append(depth_change / 10.0)

        # Has ID?
        has_id = bool(re.search(r"/{id}|\d+", candidate["endpoint"]))
        features.append(1.0 if has_id else 0.0)

        # Resource in recent?
        recent_resources = [self._get_resource(h["endpoint"]) for h in history[-3:]]
        in_recent = cand_res in recent_resources
        features.append(1.0 if in_recent else 0.0)

        return features

    def _prompt_features(self, candidate: Dict, prompt: Optional[str]) -> List[float]:
        """Prompt alignment features"""
        features = []

        # Has prompt?
        features.append(1.0 if prompt else 0.0)

        if not prompt:
            features.extend([0.0] * 5)
            return features

        prompt_lower = prompt.lower() if prompt else ""
        cand_method = self._get_method(candidate["endpoint"])

        # Verb alignment
        verb_scores = {
            "GET": ["view", "get", "list", "show", "fetch"],
            "POST": ["create", "add", "new", "make"],
            "PUT": ["update", "edit", "modify"],
            "DELETE": ["delete", "remove", "destroy"],
        }

        score = 0.0
        if cand_method in verb_scores:
            matches = sum(1 for v in verb_scores[cand_method] if v in prompt_lower)
            score = min(matches / len(verb_scores[cand_method]), 1.0)
        features.append(score)

        # Resource in prompt?
        resource = self._get_resource(candidate["endpoint"])
        features.append(1.0 if resource.lower() in prompt_lower else 0.0)

        # Prompt complexity
        features.append(len(prompt.split()) / 20.0 if prompt else 0.0)

        # Has action words?
        action_words = ["create", "update", "delete", "add", "remove", "edit"]
        has_action = any(word in prompt_lower for word in action_words)
        features.append(1.0 if has_action else 0.0)

        # Unsafe?
        is_destructive = "DELETE" in candidate["endpoint"]
        has_permission = "delete" in prompt_lower or "remove" in prompt_lower
        unsafe = is_destructive and not has_permission
        features.append(1.0 if unsafe else 0.0)

        return features

    def _pattern_features(self, history: List[Dict], candidate: Dict) -> List[float]:
        """Common API patterns"""
        features = []

        last = history[-1]
        last_method = self._get_method(last["endpoint"])
        cand_method = self._get_method(candidate["endpoint"])

        # List to detail?
        list_to_detail = (
            last_method == "GET"
            and cand_method == "GET"
            and "/{id}" not in last["endpoint"]
            and "/{id}" in candidate["endpoint"]
        )
        features.append(1.0 if list_to_detail else 0.0)

        # Detail to update?
        detail_to_update = (
            last_method == "GET"
            and cand_method in ["PUT", "PATCH"]
            and "/{id}" in last["endpoint"]
        )
        features.append(1.0 if detail_to_update else 0.0)

        # Create to view?
        create_to_view = last_method == "POST" and cand_method == "GET"
        features.append(1.0 if create_to_view else 0.0)

        # Update to view?
        update_to_view = last_method in ["PUT", "PATCH"] and cand_method == "GET"
        features.append(1.0 if update_to_view else 0.0)

        # Safe operation?
        safe = cand_method in ["GET", "POST"]
        features.append(1.0 if safe else 0.0)

        return features

    def _get_method(self, endpoint: str) -> str:
        """Extract HTTP method"""
        return endpoint.split()[0] if endpoint else "GET"

    def _get_resource(self, endpoint: str) -> str:
        """Extract base resource"""
        parts = endpoint.split("/")
        return parts[1] if len(parts) > 1 else ""
