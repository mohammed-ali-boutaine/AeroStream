from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator # type: ignore
import requests
import psycopg2
from psycopg2.extras import execute_values
import re



# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# Configuration
API_BASE_URL = 'http://host.docker.internal:8000'  
DB_CONFIG = {
    'host': 'postgres_backend',  
    'database': 'backend_db',
    'user': 'ali',
    'password': 'root'
}


def fetch_data_from_api(**context):
    """
    Task 1: Fetch fake tweets from the API in micro-batches
    """
    try:
        batch_size = 20  # Micro-batch size
        url = f"{API_BASE_URL}/fake-tweets?batch_size={batch_size}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        tweets = response.json()
        
        # Push data to XCom for next task
        context['task_instance'].xcom_push(key='raw_tweets', value=tweets)
        
        return f"Fetched {len(tweets)} tweets in micro-batch"

    except Exception as e:
        print("API request failed: {e}")
        raise


def clean_text(text):
    """
    Clean and preprocess text data
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (keep the text)
    text = re.sub(r'#', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def process_text_data(**context):
    """
    Task 2: Process the text data
    - Clean and preprocess text for sentiment prediction
    - Prepare data for storage
    """
    try:
        # Pull raw tweets from XCom
        tweets = context['task_instance'].xcom_pull(key='raw_tweets', task_ids='fetch_data_from_api')
        
        if not tweets:
            raise ValueError("No tweets received from API")
        
        
        processed_data = []
        for tweet in tweets:
            # Clean the text
            clean_tweet_text = clean_text(tweet.get('text', ''))
            
            # Extract and process fields
            processed_tweet = {
                'airline': tweet.get('airline'),
                'airline_sentiment': tweet.get('airline_sentiment'),  # From API (simulated prediction)
                'negativereason': tweet.get('negativereason'),
                'tweet_created': tweet.get('tweet_created'),
                'text': tweet.get('text', ''),
                'clean_text': clean_tweet_text,
                'processed_at': datetime.now().isoformat()
            }
            processed_data.append(processed_tweet)
        
        
        # Push processed data to XCom
        context['task_instance'].xcom_push(key='processed_tweets', value=processed_data)
        
        return f"Processed {len(processed_data)} tweets"
    
    except Exception as e:
        print(f"Error in process_text_data: {e}")
        raise


def store_in_database(**context):
    """
    Task 3: Store processed sentiment predictions in PostgreSQL database
    """
    try:
        # Pull processed tweets from XCom
        processed_tweets = context['task_instance'].xcom_pull(
            key='processed_tweets', 
            task_ids='process_text_data'
        )
        
                
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Prepare data for bulk insert
        insert_query = """
        INSERT INTO airline_tweets 
        (airline_sentiment, negativereason, airline, text, tweet_created, clean_text)
        VALUES %s
        """
        
        values = [
            (
                tweet['airline_sentiment'],
                tweet['negativereason'] if tweet['negativereason'] else None,
                tweet['airline'],
                tweet['text'],
                tweet['tweet_created'],
                tweet['clean_text']
            )
            for tweet in processed_tweets
        ]
        
        # Bulk insert
        execute_values(cursor, insert_query, values)
        conn.commit()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM airline_tweets")
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        
        return f"Stored {len(processed_tweets)} tweets. Total: {total_count}"
    
    except Exception as e:

        print(f"Error in store_in_database: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        raise



# Define the DAG
with DAG(
    'airline_sentiment_etl_pipeline',
    default_args=default_args,
    description='ETL: Fetch tweets in micro-batches, preprocess, predict sentiment, and store in PostgreSQL',
    schedule_interval='@hourly',  # Run every hour for micro-batch processing
    catchup=False,
    tags=['etl', 'sentiment-analysis', 'airline', 'micro-batch'],
) as dag:
    
    # Task 1: Fetch data from API
    fetch_data = PythonOperator(
        task_id='fetch_data_from_api',
        python_callable=fetch_data_from_api,
        provide_context=True,
    )
    
    # Task 2: Process text data
    process_data = PythonOperator(
        task_id='process_text_data',
        python_callable=process_text_data,
        provide_context=True,
    )
    
    # Task 3: Store in database
    store_data = PythonOperator(
        task_id='store_in_database',
        python_callable=store_in_database,
        provide_context=True,
    )
    
    # Define task dependencies
    fetch_data >> process_data >> store_data 