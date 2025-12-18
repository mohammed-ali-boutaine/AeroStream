# Airline Sentiment API

REST API for airline sentiment analysis using machine learning models.

## Features

- ✅ Single text prediction
- ✅ Batch prediction
- ✅ Sentiment classification (negative, neutral, positive)
- ✅ Confidence scores
- ✅ Probability distribution for all classes
- ✅ FastAPI with interactive docs

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure models are trained:**
   - Run notebooks 1-4 to train and save models
   - Models should be in `../models/best_model.pkl`
   - Label encoder (if needed) in `../models/label_encoder.pkl`

## Running the API

### Development Mode

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Root Endpoint
```bash
GET /
```

### 2. Health Check
```bash
GET /health
```

### 3. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "@VirginAmerica plus you've added commercials to the experience... tacky."
}
```

**Response:**
```json
{
  "text": "@VirginAmerica plus you've added commercials to the experience... tacky.",
  "clean_text": "VirginAmerica plus you've added commercials to the experience tacky.",
  "predicted_sentiment": "negative",
  "confidence": 0.8523,
  "probabilities": {
    "negative": 0.8523,
    "neutral": 0.1234,
    "positive": 0.0243
  }
}
```

### 4. Batch Prediction
```bash
POST /batch-predict
Content-Type: application/json

{
  "texts": [
    "@united amazing flight! Great service!",
    "@SouthwestAir flight delayed again"
  ]
}
```

**Response:**
```json
{
  "count": 2,
  "results": [
    {
      "text": "@united amazing flight! Great service!",
      "predicted_sentiment": "positive",
      "confidence": 0.9123,
      "probabilities": {...}
    },
    {
      "text": "@SouthwestAir flight delayed again",
      "predicted_sentiment": "negative",
      "confidence": 0.7845,
      "probabilities": {...}
    }
  ]
}
```

## Interactive Documentation

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Testing

Run the test script:
```bash
python test_api.py
```

Or use curl:
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "@VirginAmerica worst flight ever!"}'

# Batch prediction
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great service!", "Flight delayed", "Average experience"]}'
```

## Project Structure

```
backend/
├── main.py              # FastAPI application
├── services.py          # Prediction service
├── requirements.txt     # Python dependencies
├── test_api.py         # API test script
└── README.md           # This file
```

## Model Information

- **Embeddings:** SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
- **Classifier:** Best model from notebook 4 (Logistic Regression, Random Forest, XGBoost, or MLP)
- **Classes:** negative, neutral, positive

## Troubleshooting

### Model not found
```
FileNotFoundError: ../models/best_model.pkl
```
**Solution:** Run notebook 4 (Modeling) to train and save the model.

### Service unavailable (503)
```
Prediction service not available
```
**Solution:** Check that the model loaded successfully in the console output.

## Next Steps

- Deploy to production (Docker, Cloud)
- Add authentication
- Add rate limiting
- Add request logging
- Create frontend interface
