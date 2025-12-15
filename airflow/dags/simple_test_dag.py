from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import psycopg2
from psycopg2 import sql

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
}

# Define the DAG
dag = DAG(
    'simple_test_dag',
    default_args=default_args,
    description='A simple test DAG for Airflow',
    schedule_interval=timedelta(seconds=30),
    catchup=False,
)

# Python function for a task
def insert_hello_to_db():
    """Insert 'hello' into PostgreSQL database"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host='postgres',
            database='airflow',
            user='airflow',
            password='airflow'
        )
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        create_table = """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            message VARCHAR(255),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table)
        
        # Insert 'hello' into the table
        insert_query = "INSERT INTO messages (message) VALUES (%s)"
        cursor.execute(insert_query, ('hello',))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print('Successfully inserted "hello" into PostgreSQL')
        return 'Data inserted successfully'
    except Exception as e:
        print(f'Error inserting data: {str(e)}')
        raise

def insert_date_to_db():
    """Insert current date into PostgreSQL database"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host='postgres',
            database='airflow',
            user='airflow',
            password='airflow'
        )
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        create_table = """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            message VARCHAR(255),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table)
        
        # Insert current date into the table
        insert_query = "INSERT INTO messages (message) VALUES (%s)"
        current_date = f'Current date: {datetime.now()}'
        cursor.execute(insert_query, (current_date,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print('Successfully inserted date into PostgreSQL')
        return 'Date inserted successfully'
    except Exception as e:
        print(f'Error inserting date: {str(e)}')
        raise

# Define tasks
task_1 = PythonOperator(
    task_id='insert_hello',
    python_callable=insert_hello_to_db,
    dag=dag,
)

task_2 = BashOperator(
    task_id='bash_task',
    bash_command='echo "Running bash command in Airflow"',
    dag=dag,
)

task_3 = PythonOperator(
    task_id='insert_date',
    python_callable=insert_date_to_db,
    dag=dag,
)

# Set task dependencies
task_1 >> task_2 >> task_3
