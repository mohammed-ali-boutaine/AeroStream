"""SQL queries for Streamlit dashboard"""

# KPI queries
KPI_SUMMARY = """
    SELECT 
        COUNT(*) as total_tweets,
        COUNT(DISTINCT airline) as total_airlines,
        SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) as negative_tweets,
        ROUND(
            100.0 * SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) / COUNT(*), 
            2
        ) as negative_percentage,
        SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) as positive_tweets,
        SUM(CASE WHEN airline_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_tweets
    FROM airline_tweets
"""

SENTIMENT_DISTRIBUTION = """
    SELECT airline_sentiment, COUNT(*) as count
    FROM airline_tweets
    GROUP BY airline_sentiment
    ORDER BY count DESC
"""

# Airline queries
TWEETS_BY_AIRLINE = """
    SELECT 
        airline,
        COUNT(*) as tweet_count,
        SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN airline_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
        SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
        ROUND(
            100.0 * SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) / COUNT(*),
            2
        ) as satisfaction_rate
    FROM airline_tweets
    GROUP BY airline
    ORDER BY tweet_count DESC
"""

# Negative reasons
NEGATIVE_REASONS = """
    SELECT 
        negativereason,
        COUNT(*) as count
    FROM airline_tweets
    WHERE negativereason IS NOT NULL AND negativereason != ''
    GROUP BY negativereason
    ORDER BY count DESC
    LIMIT 10
"""

# Recent tweets
RECENT_TWEETS = """
    SELECT 
        id, airline, airline_sentiment, negativereason,
        text, clean_text, tweet_created, created_at
    FROM airline_tweets
    ORDER BY created_at DESC
    LIMIT :limit
"""

# Time series
TIME_SERIES = """
    SELECT 
        DATE_TRUNC('hour', created_at) as hour,
        COUNT(*) as tweet_count,
        SUM(CASE WHEN airline_sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN airline_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
        SUM(CASE WHEN airline_sentiment = 'negative' THEN 1 ELSE 0 END) as negative
    FROM airline_tweets
    GROUP BY hour
    ORDER BY hour DESC
    LIMIT 100
"""

# Airline specific stats
AIRLINE_STATS = """
    SELECT 
        airline,
        airline_sentiment,
        COUNT(*) as count,
        ROUND(AVG(LENGTH(clean_text)), 0) as avg_text_length
    FROM airline_tweets
    WHERE airline = :airline
    GROUP BY airline, airline_sentiment
"""

# Top complaints by airline
TOP_COMPLAINTS = """
    SELECT 
        airline,
        negativereason,
        COUNT(*) as complaint_count
    FROM airline_tweets
    WHERE airline_sentiment = 'negative' 
        AND negativereason IS NOT NULL
        AND negativereason != ''
    GROUP BY airline, negativereason
    ORDER BY complaint_count DESC
    LIMIT 15
"""
