"""
OpenRouter LLM Client for API Candidate Generation

Generates likely next API calls using LLM based on:
- OpenAPI specification
- User's recent API call history
- Optional natural language prompt

Optimized for speed and accuracy in API prediction scenarios.
"""

import httpx
import json
import os
import asyncio
import yaml
from typing import List, Dict, Optional, Any
import re
from urllib.parse import urlparse


class OpenRouterClient:
    """OpenRouter API client for generating API call candidates"""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "openai/gpt-3.5-turbo"
    ):
        """Initialize OpenRouter client

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model to use for generation (default: gpt-3.5-turbo for speed)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key parameter."
            )

        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model

        # Request settings optimized for speed
        self.timeout = 10.0
        self.max_tokens = 1500  # Limit for faster responses
        self.temperature = 0.7  # Balanced creativity

    async def generate_candidates(
        self,
        spec_url: str,
        events: List[Dict],
        prompt: Optional[str] = None,
        n: int = 15,
    ) -> List[Dict]:
        """Generate API call candidates

        Args:
            spec_url: URL to OpenAPI specification
            events: Recent API call history [{"endpoint": "GET /users", "params": {}}, ...]
            prompt: Optional user intent prompt
            n: Number of candidates to generate

        Returns:
            List of candidates: [{"endpoint": "POST /users", "params": {...}, "reasoning": "..."}, ...]
        """

        try:
            # Load and parse OpenAPI spec
            endpoints = await self._load_api_endpoints(spec_url)

            # Build LLM prompt
            llm_prompt = self._build_prompt(endpoints, events, prompt, n)

            # Call OpenRouter
            candidates = await self._call_openrouter(llm_prompt, n)

            # Validate and clean candidates
            return self._validate_candidates(candidates, endpoints)

        except Exception as e:
            print(f"Error generating candidates: {e}")
            # Return fallback candidates
            return self._generate_fallback_candidates(events, n)

    async def _load_api_endpoints(self, spec_url: str) -> List[str]:
        """Load and extract endpoints from OpenAPI spec"""

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(spec_url)
                response.raise_for_status()

            # Parse based on file extension
            if spec_url.endswith((".yaml", ".yml")):
                spec = yaml.safe_load(response.text)
            else:
                spec = json.loads(response.text)

            return self._extract_endpoints_from_spec(spec)

        except Exception as e:
            print(f"Warning: Failed to load OpenAPI spec from {spec_url}: {e}")
            # Return common REST endpoints as fallback
            return self._get_fallback_endpoints()

    def _extract_endpoints_from_spec(self, spec: Dict) -> List[str]:
        """Extract all endpoints from OpenAPI specification"""

        endpoints = []

        # Handle OpenAPI 3.0+ format
        if "paths" in spec:
            for path, methods in spec.get("paths", {}).items():
                for method in ["get", "post", "put", "patch", "delete"]:
                    if method in methods:
                        endpoints.append(f"{method.upper()} {path}")

        # Limit to most relevant endpoints (avoid overwhelming LLM)
        return endpoints[:100]  # Top 100 endpoints

    def _get_fallback_endpoints(self) -> List[str]:
        """Generate generic REST endpoints when spec loading fails"""

        return [
            "GET /api/users",
            "POST /api/users",
            "GET /api/users/{id}",
            "PUT /api/users/{id}",
            "DELETE /api/users/{id}",
            "GET /api/orders",
            "POST /api/orders",
            "GET /api/orders/{id}",
            "PUT /api/orders/{id}",
            "DELETE /api/orders/{id}",
            "GET /api/products",
            "POST /api/products",
            "GET /api/products/{id}",
            "PUT /api/products/{id}",
            "DELETE /api/products/{id}",
            "GET /api/customers",
            "POST /api/customers",
            "GET /api/customers/{id}",
            "PUT /api/customers/{id}",
            "DELETE /api/customers/{id}",
            "GET /api/payments",
            "POST /api/payments",
            "GET /api/payments/{id}",
            "GET /api/invoices",
            "POST /api/invoices",
            "GET /api/invoices/{id}",
            "PUT /api/invoices/{id}",
            "DELETE /api/invoices/{id}",
        ]

    def _build_prompt(
        self, endpoints: List[str], events: List[Dict], prompt: Optional[str], n: int
    ) -> str:
        """Build optimized prompt for LLM"""

        # Format recent events
        recent_events = events[-5:] if len(events) > 5 else events
        history_text = "\n".join([f"- {event['endpoint']}" for event in recent_events])

        # Select most relevant endpoints (avoid token overflow)
        relevant_endpoints = self._select_relevant_endpoints(endpoints, recent_events)
        endpoints_text = "\n".join([f"- {ep}" for ep in relevant_endpoints])

        # User intent
        intent_text = (
            f'User intent: "{prompt}"' if prompt else "No specific user intent provided"
        )

        system_prompt = """You are an expert API prediction assistant. Generate the most likely next API calls based on common REST patterns and user behavior."""

        user_prompt = f"""
API ENDPOINTS AVAILABLE:
{endpoints_text}

RECENT API CALL HISTORY:
{history_text}

{intent_text}

TASK: Generate exactly {n} most likely next API calls the user will make.

RULES:
1. Follow logical REST API patterns (list→detail, create→view, update→view, etc.)
2. Consider the user's intent if provided
3. Prioritize safe operations (GET, POST) over destructive ones (DELETE)
4. Use realistic parameter values when needed
5. Provide brief reasoning for each suggestion

OUTPUT FORMAT (JSON only, no other text):
[
  {{"endpoint": "GET /api/resource", "params": {{}}, "reasoning": "Brief explanation"}},
  {{"endpoint": "POST /api/resource", "params": {{"key": "value"}}, "reasoning": "Brief explanation"}}
]

Generate {n} candidates now:"""

        return user_prompt

    def _select_relevant_endpoints(
        self, endpoints: List[str], recent_events: List[Dict]
    ) -> List[str]:
        """Select most relevant endpoints to avoid token overflow"""

        if not recent_events:
            return endpoints[:30]  # Just return first 30 if no history

        # Extract resources from recent events
        recent_resources = set()
        for event in recent_events:
            resource = self._extract_resource_from_endpoint(event["endpoint"])
            if resource:
                recent_resources.add(resource)

        # Prioritize endpoints related to recent resources
        relevant = []
        other = []

        for endpoint in endpoints:
            endpoint_resource = self._extract_resource_from_endpoint(endpoint)
            if endpoint_resource in recent_resources:
                relevant.append(endpoint)
            else:
                other.append(endpoint)

        # Return relevant first, then others, limited to 30 total
        return (relevant + other)[:30]

    def _extract_resource_from_endpoint(self, endpoint: str) -> str:
        """Extract resource name from endpoint"""

        # Handle "METHOD /path/resource" format
        parts = endpoint.split("/")
        if len(parts) >= 2:
            # Get first non-empty path segment
            for part in parts[1:]:
                if part and "{" not in part:  # Skip path parameters
                    return part
        return ""

    async def _call_openrouter(self, prompt: str, n: int) -> List[Dict]:
        """Make request to OpenRouter API"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert API prediction assistant. Generate the most likely next API calls based on common REST patterns and user behavior.",
            },
            {"role": "user", "content": prompt},
        ]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "API Predictor Service",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": 0.9,
                },
            )

        if response.status_code != 200:
            raise Exception(
                f"OpenRouter API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSON response
        try:
            # Try direct JSON parsing
            candidates = json.loads(content)
            return candidates if isinstance(candidates, list) else []

        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                try:
                    candidates = json.loads(json_match.group())
                    return candidates if isinstance(candidates, list) else []
                except:
                    pass

            # Fallback: parse line by line
            return self._parse_fallback_response(content, n)

    def _parse_fallback_response(self, content: str, n: int) -> List[Dict]:
        """Parse non-JSON response as fallback"""

        candidates = []
        lines = content.split("\n")

        for line in lines:
            # Look for endpoint patterns
            endpoint_match = re.search(
                r"(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s,]+)", line, re.I
            )
            if endpoint_match:
                endpoint = (
                    f"{endpoint_match.group(1).upper()} {endpoint_match.group(2)}"
                )
                candidates.append(
                    {
                        "endpoint": endpoint,
                        "params": {},
                        "reasoning": "Parsed from LLM response",
                    }
                )

                if len(candidates) >= n:
                    break

        return candidates[:n]

    def _validate_candidates(
        self, candidates: List[Dict], valid_endpoints: List[str]
    ) -> List[Dict]:
        """Validate and clean up candidates"""

        validated = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            endpoint = candidate.get("endpoint", "")
            if not endpoint:
                continue

            # Clean up endpoint format
            endpoint = self._normalize_endpoint_format(endpoint)

            # Validate endpoint exists in spec (or is reasonable)
            if self._is_valid_endpoint(endpoint, valid_endpoints):
                validated.append(
                    {
                        "endpoint": endpoint,
                        "params": candidate.get("params", {}),
                        "reasoning": candidate.get("reasoning", "LLM suggestion")[
                            :100
                        ],  # Limit reasoning length
                    }
                )

        return validated

    def _normalize_endpoint_format(self, endpoint: str) -> str:
        """Normalize endpoint to standard format"""

        # Ensure proper METHOD /path format
        parts = endpoint.strip().split()
        if len(parts) >= 2:
            method = parts[0].upper()
            path = " ".join(parts[1:])

            # Ensure path starts with /
            if not path.startswith("/"):
                path = "/" + path

            return f"{method} {path}"

        return endpoint

    def _is_valid_endpoint(self, endpoint: str, valid_endpoints: List[str]) -> bool:
        """Check if endpoint is valid/reasonable"""

        # Basic format check
        if not re.match(r"^(GET|POST|PUT|PATCH|DELETE)\s+/\S+", endpoint):
            return False

        # Check against known endpoints (with parameter substitution)
        normalized_endpoint = re.sub(r"/\{[^}]+\}", "/{id}", endpoint)
        normalized_valid = [
            re.sub(r"/\{[^}]+\}", "/{id}", ep) for ep in valid_endpoints
        ]

        if normalized_endpoint in normalized_valid:
            return True

        # Allow reasonable-looking endpoints even if not in spec
        return True  # Be permissive for now

    def _generate_fallback_candidates(self, events: List[Dict], n: int) -> List[Dict]:
        """Generate simple fallback candidates when LLM fails"""

        candidates = []

        if events:
            # Generate based on last event
            last_endpoint = events[-1].get("endpoint", "")
            last_method, last_path = self._parse_endpoint(last_endpoint)

            # Common follow-up patterns
            patterns = [
                ("GET", last_path),  # Refresh current
                ("GET", f"{last_path}/details"),  # Get details
                ("PUT", last_path),  # Update
                ("POST", "/".join(last_path.split("/")[:-1])),  # Create in parent
                ("GET", "/".join(last_path.split("/")[:-1])),  # List parent
            ]

            for method, path in patterns:
                if len(candidates) >= n:
                    break
                candidates.append(
                    {
                        "endpoint": f"{method} {path}",
                        "params": {},
                        "reasoning": "Fallback pattern-based suggestion",
                    }
                )

        # Fill with generic endpoints
        generic = [
            {
                "endpoint": "GET /api/users",
                "params": {},
                "reasoning": "Common starting point",
            },
            {
                "endpoint": "GET /api/dashboard",
                "params": {},
                "reasoning": "Common navigation",
            },
            {"endpoint": "GET /health", "params": {}, "reasoning": "Health check"},
        ]

        candidates.extend(generic)
        return candidates[:n]

    def _parse_endpoint(self, endpoint: str) -> tuple:
        """Parse endpoint into method and path"""
        parts = endpoint.split(" ", 1)
        method = parts[0] if len(parts) > 0 else "GET"
        path = parts[1] if len(parts) > 1 else "/api"
        return method, path

    async def test_connection(self) -> bool:
        """Test OpenRouter API connection"""

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "http://localhost:8000",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                    },
                )
                return response.status_code == 200

        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# Quick test function
async def test_client():
    """Test the OpenRouter client with sample data"""

    client = OpenRouterClient()

    # Test connection
    print("🔗 Testing connection...")
    connected = await client.test_connection()
    print(f"Connection: {'✅ OK' if connected else '❌ Failed'}")

    if not connected:
        print("⚠️ Check your OPENROUTER_API_KEY")
        return

    # Test candidate generation
    print("\n🤖 Testing candidate generation...")

    events = [{"endpoint": "GET /users"}, {"endpoint": "GET /users/123"}]

    spec_url = "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.yaml"
    prompt = "update user information"

    try:
        candidates = await client.generate_candidates(
            spec_url=spec_url, events=events, prompt=prompt, n=5
        )

        print(f"\n✅ Generated {len(candidates)} candidates:")
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. {candidate['endpoint']}")
            print(f"   Params: {candidate.get('params', {})}")
            print(f"   Why: {candidate.get('reasoning', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_client())

