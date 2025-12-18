"""Airline Sentiment Prediction Service."""
import re
from pathlib import Path
import numpy as np
import pickle
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AirlineSentimentService:
    """Service for airline sentiment prediction."""
    
    def __init__(self, model_path="../models/best_model.pkl", encoder_path="../models/label_encoder.pkl"):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.model = None
        self.label_encoder = None
        self.sentence_transformer = None
        self._load_model()
        self._load_label_encoder()
        self._load_sentence_transformer()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_label_encoder(self):
        """Load label encoder if it exists (for XGBoost/MLP models)."""
        try:
            if self.encoder_path.exists():
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                logger.warning("No label encoder found (model may use string labels directly)")
        except Exception as e:
            logger.warning(f"Failed to load label encoder: {e}")
    
    def _load_sentence_transformer(self):
        """Load sentence transformer for embeddings."""
        try:
            self.sentence_transformer = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for prediction."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, text: str) -> dict:
        """
        Predict airline sentiment.
        
        Returns:
            dict: {
                "text": original text,
                "clean_text": preprocessed text,
                "predicted_sentiment": sentiment label,
                "confidence": max probability,
                "probabilities": all class probabilities
            }
        """
        # Preprocess
        clean_text = self.preprocess_text(text)
        
        # Generate embedding
        embedding = self.sentence_transformer.encode([clean_text])[0]
        embedding = embedding.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(embedding)[0]
        probabilities = self.model.predict_proba(embedding)[0]
        
        # Decode label if using label encoder
        if self.label_encoder is not None:
            if isinstance(prediction, (int, np.integer)):
                predicted_sentiment = self.label_encoder.inverse_transform([prediction])[0]
            else:
                predicted_sentiment = str(prediction)
        else:
            predicted_sentiment = str(prediction)
        
        # Get class labels
        if hasattr(self.model, 'classes_'):
            classes = self.model.classes_
            if self.label_encoder is not None and all(isinstance(c, (int, np.integer)) for c in classes):
                classes = self.label_encoder.inverse_transform(classes)
        else:
            classes = ['negative', 'neutral', 'positive']
        
        # Build probabilities dict
        proba_dict = {
            str(cls): float(prob) 
            for cls, prob in zip(classes, probabilities)
        }
        
        confidence = float(np.max(probabilities))
        
        return {
            "text": text,
            "clean_text": clean_text,
            "predicted_sentiment": predicted_sentiment,
            "confidence": confidence,
            "probabilities": proba_dict
        }
    
    def batch_predict(self, texts: list) -> list:
        """Predict sentiments for multiple texts."""
        results = []
        for text in texts:
            try:
                results.append(self.predict(text))
            except Exception as e:
                results.append({
                    "text": text,
                    "error": str(e)
                })
        return results
