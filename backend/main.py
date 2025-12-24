from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
import time
from typing import List
from services import AirlineSentimentService, FakeTweetService, Tweet
from schemas.prediction import BatchPredictionRequest, PredictionRequest, PredictionResponse
from database.database import test_connection



app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="Predict airline sentiment (negative, neutral, positive) using ML models",
    version="1.0.0"
)

# cors config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init services
try:
    predictor = AirlineSentimentService()
    print("Prediction service initialized successfully")
except Exception as e:
    print(f"Failed to initialize prediction service: {e}")
    predictor = None

try:
    faker = FakeTweetService()
    print("Fake tweet service initialized successfully")
except Exception as e:
    print(f"Failed to initialize fake tweet service: {e}")
    faker = None


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
    db_ok, db_count = test_connection()
    return {
        "status": "healthy",
        "services": {
            "predictor": predictor is not None,
            "faker": faker is not None,
            "database": db_ok
        },
        "tweets": db_count if db_ok else 0
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
        result = predictor.predict(req.text)
        print(f"Prediction result: {result['predicted_sentiment']} (confidence: {result['confidence']:.2f})")
        return PredictionResponse(**result)
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
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
        results = predictor.batch_predict(req.texts)
        print(f"Batch prediction completed: {len(results)} results")
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        print(f"Batch prediction failed: {str(e)}")
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
        print(f"Generated {len(tweets)} fake tweets")
        return tweets
    except Exception as e:
        print(f"Fake tweet generation failed: {str(e)}")
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

        # generate fake tweet and predict
        tweet = faker.generate_tweet()
        
        result = predictor.predict(tweet.text)
        
        print(f"Generated and predicted fake tweet: {result['predicted_sentiment']}")
        
        return {
            "fake_tweet": tweet,
            "prediction": result
        }
    except Exception as e:
        print(f"Fake tweet prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


from sqlalchemy import create_engine,text
import json
@app.get("/test")
def get():

    db_url =  "postgresql://ali:root@postgres_backend:5432/backend_db"

    engine = create_engine(db_url)

    with engine.connect() as conn :
        rslt = conn.execute(text("select * from airline_tweets")).fetchall()

        print(rslt)

        return {
            "data" : "hy"
        }