"""Test script for the Airline Sentiment API."""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_single_prediction():
    """Test single prediction."""
    print("\n" + "="*60)
    print("Testing /predict endpoint")
    print("="*60)
    
    test_cases = [
        "@VirginAmerica plus you've added commercials to the experience... tacky.",
        "@united amazing flight! Great service and friendly crew!",
        "@SouthwestAir my flight was cancelled for the third time"
    ]
    
    for text in test_cases:
        print(f"\nText: {text}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Sentiment: {result['predicted_sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {json.dumps(result['probabilities'], indent=2)}")
        else:
            print(f"Error: {response.text}")


def test_batch_prediction():
    """Test batch prediction."""
    print("\n" + "="*60)
    print("Testing /batch-predict endpoint")
    print("="*60)
    
    texts = [
        "@VirginAmerica plus you've added commercials to the experience... tacky.",
        "@united amazing flight! Great service!",
        "@SouthwestAir flight delayed again",
        "Worst experience ever. Never flying with them again.",
        "Pretty average flight. Nothing special."
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch-predict",
        json={"texts": texts}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total predictions: {result['count']}\n")
        for i, pred in enumerate(result['results'], 1):
            print(f"{i}. Text: {pred['text'][:60]}...")
            print(f"   Sentiment: {pred['predicted_sentiment']} (confidence: {pred['confidence']:.4f})")
    else:
        print(f"Error: {response.text}")


def test_fake_tweets():
    """Test fake tweet generation."""
    print("\n" + "="*60)
    print("Testing /fake-tweets endpoint")
    print("="*60)
    
    batch_sizes = [5, 15]
    
    for batch_size in batch_sizes:
        print(f"\nGenerating {batch_size} fake tweets...")
        response = requests.get(f"{BASE_URL}/fake-tweets?batch_size={batch_size}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            tweets = response.json()
            print(f"Generated: {len(tweets)} tweets\n")
            for i, tweet in enumerate(tweets[:3], 1):  # Show first 3
                print(f"{i}. Airline: {tweet['airline']}")
                print(f"   Text: {tweet['text'][:70]}...")
                print(f"   Confidence: {tweet['airline_sentiment_confidence']:.3f}")
                if tweet['negativereason']:
                    print(f"   Negative Reason: {tweet['negativereason']}")
        else:
            print(f"Error: {response.text}")


def test_predict_fake():
    """Test generating and predicting fake tweet."""
    print("\n" + "="*60)
    print("Testing /predict-fake endpoint")
    print("="*60)
    
    for i in range(3):
        print(f"\nTest {i+1}:")
        response = requests.post(f"{BASE_URL}/predict-fake")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            fake_tweet = result['fake_tweet']
            prediction = result['prediction']
            
            print(f"Airline: {fake_tweet['airline']}")
            print(f"Tweet: {fake_tweet['text'][:70]}...")
            print(f"Generated Confidence: {fake_tweet['airline_sentiment_confidence']:.3f}")
            print(f"Predicted Sentiment: {prediction['predicted_sentiment']}")
            print(f"Prediction Confidence: {prediction['confidence']:.4f}")
        else:
            print(f"Error: {response.text}")


if __name__ == "__main__":
    print("="*60)
    print("AIRLINE SENTIMENT API TESTS")
    print("="*60)
    print("Make sure the API is running: uvicorn main:app --reload")
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_fake_tweets()
        test_predict_fake()
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API. Is it running?")
        print("Start it with: uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
