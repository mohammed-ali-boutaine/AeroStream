#!/bin/bash
set -e

echo "Waiting for postgres..."
while ! nc -z postgres_airflow 5432; do
  sleep 1
done
echo "PostgreSQL started"

echo "Initializing Airflow database..."
airflow db init || airflow db migrate || true

echo "Creating admin user..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com || echo "User already exists"

echo "Starting Airflow webserver and scheduler..."
airflow scheduler &
exec airflow webserver
