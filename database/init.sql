-- PostgreSQL initialization script for airline_tweets table
-- This script runs automatically when the Postgres container starts for the first time

-- Create airline_tweets table
CREATE TABLE IF NOT EXISTS airline_tweets (
    id SERIAL PRIMARY KEY,
    airline_sentiment VARCHAR(20) NOT NULL,
    negativereason VARCHAR(100),
    airline VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    tweet_created TIMESTAMP,
    clean_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_sentiment CHECK (airline_sentiment IN ('positive', 'negative', 'neutral'))
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_airline ON airline_tweets(airline);
CREATE INDEX IF NOT EXISTS idx_sentiment ON airline_tweets(airline_sentiment);
CREATE INDEX IF NOT EXISTS idx_tweet_created ON airline_tweets(tweet_created);
CREATE INDEX IF NOT EXISTS idx_created_at ON airline_tweets(created_at);

-- Grant privileges
GRANT ALL PRIVILEGES ON TABLE airline_tweets TO ali;
GRANT ALL PRIVILEGES ON SEQUENCE airline_tweets_id_seq TO ali;
