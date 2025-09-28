# API Call Predictor

Predicts the next API call a user will make using LLM + ML hybrid approach.

## Quick Start

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
docker compose up
./demo.sh
```

## How It Works

**LLM Layer**: Generates candidate API calls from OpenAPI specs
**ML Layer**: Ranks candidates using user behavior patterns

The model looks at what you've been doing, what usually comes next, and prompt context to predict your next API call.

## Performance

- **Response time**: 560-600ms typical, 880ms cold start
- **Model**: 79%  training accuracy, 0.16ms inference, 1.1KB size
- **Known issue**: Tends to predict low confidence scores due to syntethic training data

## Data & Limitations
Used  synthetic training samples since real API logs aren't available. Ideally would use real user interaction data with privacy protection. Probably through some sort of extension. Created data using generic flows from diferent pages.

## Time Spent
~8-10 hours in total. Did an initial pure vibe code run through on Thursday to see POC, then started Sat with ai help, finished sunday

## What's Next

- Real user feedback collection
- Model personalization
- Better prompt engineering
- Production monitoring
- More involved feature engineering
