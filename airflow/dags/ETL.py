from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator # type: ignore
import requests
import psycopg2
from psycopg2.extras import execute_values


# default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# config
API_BASE_URL = 'http://host.docker.internal:8000'  
DB_CONFIG = {
    'host': 'postgres_backend',  
    'database': 'backend_db',
    'user': 'ali',
    'password': 'root'
}


def health_check_api(**context):
    try:
        url = f"{API_BASE_URL}/health"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        health_status = response.json()
        print(f"API Health Check: {health_status}")
        
        return "API is healthy and ready"
    
    except Exception as e:
        print(f"API health check failed: {e}")
        raise




def fetch_data_from_api(**context):

    try:
        batch_size = 20  
        url = f"{API_BASE_URL}/fake-tweets?batch_size={batch_size}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        tweets = response.json()
        
        context['task_instance'].xcom_push(key='raw_tweets', value=tweets)
        
        return f"Fetched {len(tweets)} tweets in micro-batch"

    except Exception as e:
        print(f"API request failed: {e}")
        raise


def store_in_database(**context):
    try:
        tweets = context['task_instance'].xcom_pull(
            key='raw_tweets', 
            task_ids='fetch_data_from_api'
        )
        
        if not tweets:
            raise ValueError("No tweets received from API")
                
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Prepare data for bulk insert
        insert_query = """
        INSERT INTO airline_tweets 
        (airline_sentiment, negativereason, airline, text, tweet_created)
        VALUES %s
        """
        
        values = [
            (
                tweet['airline_sentiment'],
                tweet['negativereason'] if tweet.get('negativereason') else None,
                tweet['airline'],
                tweet['text'],
                tweet['tweet_created']
            )
            for tweet in tweets
        ]
        
        execute_values(cursor, insert_query, values)
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM airline_tweets")
        total_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return f"Stored {len(tweets)} tweets. Total: {total_count}"
    
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
    description='ETL: Health check, fetch tweets in micro-batches, and store in PostgreSQL',
    schedule_interval='*/1 * * * *',
    catchup=False,
    tags=['etl', 'sentiment-analysis', 'airline', 'micro-batch'],
) as dag:
    
    # task 1: Health check API
    health_check = PythonOperator(
        task_id='health_check_api',
        python_callable=health_check_api,
        provide_context=True,
    )
    
    # task 2: Fetch data from API
    fetch_data = PythonOperator(
        task_id='fetch_data_from_api',
        python_callable=fetch_data_from_api,
        provide_context=True,
    )
    
    # task 3: Store in database
    store_data = PythonOperator(
        task_id='store_in_database',
        python_callable=store_in_database,
        provide_context=True,
    )
    
    # define dependencies
    health_check >> fetch_data >> store_data 