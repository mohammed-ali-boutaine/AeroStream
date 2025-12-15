# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import pickle

# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 1, 1),
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# def load_data():
#     # Load your data
#     print("Loading data...")
#     # Example: df = pd.read_csv('/opt/airflow/data/raw/dataset.csv')

# def preprocess_data():
#     print("Preprocessing data...")
#     # Your preprocessing logic

# def train_model():
#     print("Training model...")
#     # Example training logic
#     # model = RandomForestClassifier()
#     # model.fit(X_train, y_train)
#     # with open('/opt/airflow/models/model.pkl', 'wb') as f:
#     #     pickle.dump(model, f)

# def evaluate_model():
#     print("Evaluating model...")
#     # Your evaluation logic

# with DAG(
#     'ml_training_pipeline',
#     default_args=default_args,
#     description='ML model training pipeline',
#     schedule_interval='@daily',
#     catchup=False,
# ) as dag:

#     load = PythonOperator(
#         task_id='load_data',
#         python_callable=load_data,
#     )

#     preprocess = PythonOperator(
#         task_id='preprocess_data',
#         python_callable=preprocess_data,
#     )

#     train = PythonOperator(
#         task_id='train_model',
#         python_callable=train_model,
#     )

#     evaluate = PythonOperator(
#         task_id='evaluate_model',
#         python_callable=evaluate_model,
#     )

#     load >> preprocess >> train >> evaluate