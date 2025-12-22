
from pydantic import BaseModel, Field
from typing import List, Dict



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
