#!/bin/bash

# Check if server is running
if ! curl -s http://localhost:8000/health >/dev/null; then
  echo "❌ API server not running. Start with: docker compose up"
  exit 1
fi

echo "✅ Running API tests..."

# Test 1: Stripe API - Payment workflow
echo "Test 1: Stripe API"

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "stripe-user-1",
    "events": [
      {"ts": "2025-01-01T10:00:00Z", "endpoint": "GET /v1/customers", "params": {}},
      {"ts": "2025-01-01T10:01:00Z", "endpoint": "GET /v1/customers/cus_123", "params": {}},
      {"ts": "2025-01-01T10:02:00Z", "endpoint": "GET /v1/payment_methods", "params": {"customer": "cus_123"}}
    ],
    "prompt": "update their credit card information",
    "spec_url": "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.yaml",
    "k": 3
  }' | python3 -m json.tool

echo "Test 2: GitHub API"

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "github-user-1",
    "events": [
      {"ts": "2025-01-01T10:00:00Z", "endpoint": "GET /repos/{owner}/{repo}", "params": {}},
      {"ts": "2025-01-01T10:01:00Z", "endpoint": "GET /repos/{owner}/{repo}/issues", "params": {}},
      {"ts": "2025-01-01T10:02:00Z", "endpoint": "GET /repos/{owner}/{repo}/labels", "params": {}}
    ],
    "prompt": "create a new bug report",
    "spec_url": "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.yaml",
    "k": 3
  }' | python3 -m json.tool

echo ""
echo "---"
echo ""

# Test 3: Cold start scenario
echo "=== Test 3: Cold Start Scenario ==="
echo "Scenario: New user with no history"
echo "Intent: 'list all my invoices'"
echo ""

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "new-user",
    "events": [],
    "prompt": "list all my invoices",
    "spec_url": "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.yaml",
    "k": 3
  }' | python3 -m json.tool

echo ""
echo "---"
echo ""

