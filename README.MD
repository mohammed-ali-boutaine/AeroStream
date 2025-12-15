┌──────────────┐
│ Streamlit UI │  → visualize results, trigger jobs
└──────┬───────┘
       │
┌──────▼───────┐
│   Airflow    │  → orchestrates ML pipeline
│ (scheduler)  │     (train, preprocess, evaluate)
└──────┬───────┘
       │
┌──────▼────────────┐
│ Jupyter Notebook  │ → model dev & experiments
│ (training code)   │
└───────────────────┘


docker compose up -d

### To connect to PostgreSQL directly from your machine
psql -h localhost -U airflow -d airflow -p 5432

### To view logs:
docker-compose logs -f airflow
docker-compose logs -f streamlit