import re
from pathlib import Path
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


class AirlineSentimentService:
    """Service for airline sentiment prediction."""
    
    def __init__(self, model_path="./models/best_model.pkl", encoder_path="./models/label_encoder.pkl"):
        # load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        #load encoder
        self.label_encoder = None
        if Path(encoder_path).exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        self.transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def preprocess_text(self, text) :

        text = re.sub(r'<[^>]+>|http\S+|www\S+|@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def predict(self, text) :

        clean_text = self.preprocess_text(text)
        embedding = self.transformer.encode([clean_text])[0].reshape(1, -1)
        
        prediction = self.model.predict(embedding)[0]
        probabilities = self.model.predict_proba(embedding)[0]
        
        # Decode label
        if self.label_encoder and isinstance(prediction, (int, np.integer)):
            sentiment = self.label_encoder.inverse_transform([prediction])[0]
            classes = self.label_encoder.inverse_transform(self.model.classes_)
        else:
            sentiment = str(prediction)
            classes = self.model.classes_ if hasattr(self.model, 'classes_') else ['negative', 'neutral', 'positive']
        
        return {
            "text": text,
            "clean_text": clean_text,
            "predicted_sentiment": sentiment,
            "confidence": float(np.max(probabilities)),
            "probabilities": {str(c): float(p) for c, p in zip(classes, probabilities)}
        }
    
    def batch_predict(self, texts):
        """Predict sentiments for multiple texts."""
        return [self.predict(text) for text in texts]
