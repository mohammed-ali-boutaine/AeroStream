from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field
from typing import List, Dict
import time
import logging
from pathlib import Path

from services import AirlineSentimentService, FakeTweetService, Tweet

# Configure logging
Path('../logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="Predict airline sentiment (negative, neutral, positive) using ML models",
    version="1.0.0"
)

# CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
try:
    predictor = AirlineSentimentService()
    logger.info("Prediction service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize prediction service: {e}")
    predictor = None

# Initialize fake tweet service
try:
    faker = FakeTweetService()
    logger.info("Fake tweet service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize fake tweet service: {e}")
    faker = None


# Request/Response Models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Tweet or review text to analyze", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "@VirginAmerica plus you've added commercials to the experience... tacky."
            }
        }


class PredictionResponse(BaseModel):
    text: str
    clean_text: str
    predicted_sentiment: str
    confidence: float
    probabilities: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "@VirginAmerica plus you've added commercials to the experience... tacky.",
                    "@united amazing flight! Great service!",
                    "@SouthwestAir flight delayed again"
                ]
            }
        }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Airline Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "fake_tweets": "/fake-tweets",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "predictor": "loaded" if predictor else "unavailable",
        "faker": "loaded" if faker else "unavailable",
        "timestamp": time.time()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(req: PredictionRequest):
    """
    Predict sentiment for a single text.
    
    - **text**: Tweet or review text to analyze
    
    Returns sentiment (negative, neutral, positive) with confidence scores.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available. Model may not be loaded."
        )
    
    try:
        logger.info(f"Prediction request: {req.text[:50]}...")
        result = predictor.predict(req.text)
        logger.info(f"Prediction result: {result['predicted_sentiment']} (confidence: {result['confidence']:.2f})")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict_sentiment(req: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts.
    
    - **texts**: List of tweets or reviews to analyze
    
    Returns list of predictions with sentiment and confidence scores.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available. Model may not be loaded."
        )
    
    try:
        logger.info(f"Batch prediction request: {len(req.texts)} texts")
        results = predictor.batch_predict(req.texts)
        logger.info(f"Batch prediction completed: {len(results)} results")
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/fake-tweets", response_model=List[Tweet])
async def generate_fake_tweets(batch_size: int = 10):
    """
    Generate fake airline tweets for testing.
    
    - **batch_size**: Number of tweets to generate (1-100)
    
    Returns list of fake tweets with sentiment labels.
    """
    if faker is None:
        raise HTTPException(
            status_code=503,
            detail="Fake tweet service not available."
        )
    
    try:
        tweets = faker.generate_batch(batch_size)
        logger.info(f"Generated {len(tweets)} fake tweets")
        return tweets
    except Exception as e:
        logger.error(f"Fake tweet generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fake tweet generation failed: {str(e)}")


@app.post("/predict-fake")
async def predict_fake_tweet():
    """
    Generate a fake tweet and predict its sentiment.
    
    Returns both the fake tweet and the prediction.
    """
    if faker is None or predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Services not available."
        )
    
    try:
        # Generate fake tweet
        tweet = faker.generate_tweet()
        
        # Predict sentiment
        result = predictor.predict(tweet.text)
        
        logger.info(f"Generated and predicted fake tweet: {result['predicted_sentiment']}")
        
        return {
            "fake_tweet": tweet,
            "prediction": result
        }
    except Exception as e:
        logger.error(f"Fake tweet prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
