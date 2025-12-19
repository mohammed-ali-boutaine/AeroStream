from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import psycopg2
from psycopg2.extras import execute_values
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# Configuration
API_BASE_URL = 'http://host.docker.internal:8000'  # Adjust if backend is in Docker network
DB_CONFIG = {
    'host': 'postgres',
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}


def fetch_data_from_api(**context):
    """
    Task 1: Fetch fake tweets from the API
    """
    try:
        batch_size = 20  # Number of tweets to fetch
        url = f"{API_BASE_URL}/fake-tweets?batch_size={batch_size}"
        
        logger.info(f"Fetching data from API: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        tweets = response.json()
        logger.info(f"Successfully fetched {len(tweets)} tweets from API")
        
        # Push data to XCom for next task
        context['task_instance'].xcom_push(key='raw_tweets', value=tweets)
        
        return f"Fetched {len(tweets)} tweets"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in fetch_data_from_api: {e}")
        raise


def process_text_data(**context):
    """
    Task 2: Process the text data
    - Clean text
    - Extract features
    - Transform data for storage
    """
    try:
        # Pull raw tweets from XCom
        tweets = context['task_instance'].xcom_pull(key='raw_tweets', task_ids='fetch_data_from_api')
        
        if not tweets:
            raise ValueError("No tweets received from API")
        
        logger.info(f"Processing {len(tweets)} tweets")
        
        processed_data = []
        for tweet in tweets:
            # Extract and process fields
            processed_tweet = {
                'airline': tweet.get('airline'),
                'sentiment': tweet.get('airline_sentiment_confidence'),  # This seems to be confidence
                'negativereason': tweet.get('negativereason'),
                'tweet_created': tweet.get('tweet_created'),
                'text': tweet.get('text', ''),
                'text_length': len(tweet.get('text', '')),
                'has_mention': '@' in tweet.get('text', ''),
                'has_hashtag': '#' in tweet.get('text', ''),
                'processed_at': datetime.now().isoformat()
            }
            processed_data.append(processed_tweet)
        
        logger.info(f"Successfully processed {len(processed_data)} tweets")
        
        # Push processed data to XCom
        context['task_instance'].xcom_push(key='processed_tweets', value=processed_data)
        
        return f"Processed {len(processed_data)} tweets"
    
    except Exception as e:
        logger.error(f"Error in process_text_data: {e}")
        raise


def store_in_database(**context):
    """
    Task 3: Store processed data in PostgreSQL database
    """
    try:
        # Pull processed tweets from XCom
        processed_tweets = context['task_instance'].xcom_pull(
            key='processed_tweets', 
            task_ids='process_text_data'
        )
        
        if not processed_tweets:
            raise ValueError("No processed tweets received")
        
        logger.info(f"Storing {len(processed_tweets)} tweets in database")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS airline_tweets (
            id SERIAL PRIMARY KEY,
            airline VARCHAR(100),
            sentiment FLOAT,
            negativereason VARCHAR(255),
            tweet_created VARCHAR(100),
            text TEXT,
            text_length INTEGER,
            has_mention BOOLEAN,
            has_hashtag BOOLEAN,
            processed_at TIMESTAMP,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        logger.info("Table created or already exists")
        
        # Prepare data for bulk insert
        insert_query = """
        INSERT INTO airline_tweets 
        (airline, sentiment, negativereason, tweet_created, text, 
         text_length, has_mention, has_hashtag, processed_at)
        VALUES %s
        """
        
        values = [
            (
                tweet['airline'],
                tweet['sentiment'],
                tweet['negativereason'],
                tweet['tweet_created'],
                tweet['text'],
                tweet['text_length'],
                tweet['has_mention'],
                tweet['has_hashtag'],
                tweet['processed_at']
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
        
        logger.info(f"Successfully stored {len(processed_tweets)} tweets. Total in DB: {total_count}")
        
        return f"Stored {len(processed_tweets)} tweets. Total: {total_count}"
    
    except Exception as e:
        logger.error(f"Error in store_in_database: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        raise


def validate_pipeline(**context):
    """
    Task 4: Validate that data was stored correctly
    """
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_tweets,
                COUNT(DISTINCT airline) as unique_airlines,
                AVG(text_length) as avg_text_length,
                MAX(inserted_at) as last_insert
            FROM airline_tweets
        """)
        
        stats = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        validation_result = {
            'total_tweets': stats[0],
            'unique_airlines': stats[1],
            'avg_text_length': float(stats[2]) if stats[2] else 0,
            'last_insert': str(stats[3])
        }
        
        logger.info(f"Pipeline validation: {validation_result}")
        
        return validation_result
    
    except Exception as e:
        logger.error(f"Error in validate_pipeline: {e}")
        raise


# Define the DAG
with DAG(
    'api_to_database_pipeline',
    default_args=default_args,
    description='Fetch tweets from API, process text, and store in PostgreSQL',
    schedule_interval='@hourly',  # Run every hour
    catchup=False,
    tags=['api', 'text-processing', 'database', 'tweets'],
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
    
    # Task 4: Validate pipeline
    validate = PythonOperator(
        task_id='validate_pipeline',
        python_callable=validate_pipeline,
        provide_context=True,
    )
    
    # Define task dependencies
    fetch_data >> process_data >> store_data >> validate