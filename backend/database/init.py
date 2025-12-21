import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """
    Handles PostgreSQL database and table creation, data validation, and insertion.
    """
    
    def __init__(self, host='localhost', port=5432, user='ali', password='root'):
        """
        Initialize database connection parameters.
        
        Args:
            host: Database host
            port: Database port
            user: Database user
            password: Database password
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = 'backend_db'
        self.conn = None
        self.cursor = None
    
    def connect_to_postgres(self):
        """Connect to PostgreSQL server (without specific database)"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database='postgres'
            )
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.conn.cursor()
            logger.info("Connected to PostgreSQL server")
            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            return False
    
    def create_database(self):
        """Create the aerostream database if it doesn't exist"""
        try:
            # Check if database exists
            self.cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.db_name,)
            )
            exists = self.cursor.fetchone()
            
            if not exists:
                self.cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.db_name)
                ))
                logger.info(f"Database '{self.db_name}' created successfully")
            else:
                logger.info(f"Database '{self.db_name}' already exists")
            
            return True
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return False
    
    def connect_to_database(self):
        """Connect to the aerostream database"""
        try:
            if self.conn:
                self.cursor.close()
                self.conn.close()
            
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.db_name
            )
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database '{self.db_name}'")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def create_tables(self):
        """Create tables for airline sentiment data"""
        try:
            # Create tweets table
            create_table_query = """
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
            
            CREATE INDEX IF NOT EXISTS idx_airline ON airline_tweets(airline);
            CREATE INDEX IF NOT EXISTS idx_sentiment ON airline_tweets(airline_sentiment);
            CREATE INDEX IF NOT EXISTS idx_tweet_created ON airline_tweets(tweet_created);
            """
            
            self.cursor.execute(create_table_query)
            self.conn.commit()
            logger.info("Tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.conn.rollback()
            return False
    
    def validate_data(self, df):
        """
        Validate data before insertion.
        
        Args:
            df: pandas DataFrame with the data
            
        Returns:
            tuple: (is_valid, cleaned_df, validation_report)
        """
        validation_report = {
            'total_rows': len(df),
            'missing_values': {},
            'invalid_sentiments': 0,
            'date_parsing_errors': 0,
            'valid_rows': 0
        }
        
        # Check required columns
        required_cols = ['airline_sentiment', 'airline', 'text', 'tweet_created', 'clean_text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False, None, validation_report
        
        # Create a copy for cleaning
        df_clean = df.copy()
        
        # Remove unnamed index column if present
        if df_clean.columns[0].startswith('Unnamed'):
            df_clean = df_clean.iloc[:, 1:]
        
        # Check for missing values in required columns
        for col in required_cols:
            missing_count = df_clean[col].isna().sum()
            validation_report['missing_values'][col] = missing_count
            if missing_count > 0:
                logger.warning(f"Column '{col}' has {missing_count} missing values")
        
        # Validate sentiment values
        valid_sentiments = ['positive', 'negative', 'neutral']
        invalid_sentiment_mask = ~df_clean['airline_sentiment'].isin(valid_sentiments)
        validation_report['invalid_sentiments'] = invalid_sentiment_mask.sum()
        
        if validation_report['invalid_sentiments'] > 0:
            logger.warning(f"Found {validation_report['invalid_sentiments']} rows with invalid sentiments")
            df_clean = df_clean[~invalid_sentiment_mask]
        
        # Parse dates
        try:
            df_clean['tweet_created'] = pd.to_datetime(df_clean['tweet_created'])
        except Exception as e:
            logger.error(f"Error parsing dates: {e}")
            validation_report['date_parsing_errors'] = len(df_clean)
            # Try parsing with errors='coerce' to convert invalid dates to NaT
            df_clean['tweet_created'] = pd.to_datetime(df_clean['tweet_created'], errors='coerce')
            validation_report['date_parsing_errors'] = df_clean['tweet_created'].isna().sum()
        
        # Remove rows with missing required values
        df_clean = df_clean.dropna(subset=['airline_sentiment', 'airline', 'text', 'clean_text'])
        
        # Fill NaN negativereason with empty string
        df_clean['negativereason'] = df_clean['negativereason'].fillna('')
        
        validation_report['valid_rows'] = len(df_clean)
        
        # Log validation summary
        logger.info("=" * 60)
        logger.info("DATA VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total rows in file: {validation_report['total_rows']}")
        logger.info(f"Valid rows after cleaning: {validation_report['valid_rows']}")
        logger.info(f"Rows removed: {validation_report['total_rows'] - validation_report['valid_rows']}")
        
        if validation_report['missing_values']:
            logger.info("\nMissing values by column:")
            for col, count in validation_report['missing_values'].items():
                if count > 0:
                    logger.info(f"  - {col}: {count}")
        
        if validation_report['invalid_sentiments'] > 0:
            logger.info(f"\nInvalid sentiment values: {validation_report['invalid_sentiments']}")
        
        if validation_report['date_parsing_errors'] > 0:
            logger.info(f"\nDate parsing errors: {validation_report['date_parsing_errors']}")
        
        logger.info("=" * 60)
        
        is_valid = validation_report['valid_rows'] > 0
        return is_valid, df_clean, validation_report
    
    def insert_data(self, df):
        """
        Insert data into the database.
        
        Args:
            df: pandas DataFrame with validated data
            
        Returns:
            bool: Success status
        """
        try:
            # Clear existing data (optional)
            logger.info("Checking for existing data...")
            self.cursor.execute("SELECT COUNT(*) FROM airline_tweets")
            existing_count = self.cursor.fetchone()[0]
            
            if existing_count > 0:
                logger.warning(f"Found {existing_count} existing rows in the table")
                response = input("Do you want to clear existing data? (yes/no): ").strip().lower()
                if response == 'yes':
                    self.cursor.execute("TRUNCATE TABLE airline_tweets RESTART IDENTITY")
                    self.conn.commit()
                    logger.info("Existing data cleared")
            
            # Prepare insert query
            insert_query = """
            INSERT INTO airline_tweets 
            (airline_sentiment, negativereason, airline, text, tweet_created, clean_text)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            # Convert DataFrame to list of tuples
            records = []
            for _, row in df.iterrows():
                tweet_created = row['tweet_created'] if pd.notna(row['tweet_created']) else None
                records.append((
                    row['airline_sentiment'],
                    row['negativereason'] if row['negativereason'] else None,
                    row['airline'],
                    row['text'],
                    tweet_created,
                    row['clean_text']
                ))
            
            # Batch insert
            logger.info(f"Inserting {len(records)} records...")
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.cursor.executemany(insert_query, batch)
                self.conn.commit()
                total_inserted += len(batch)
                logger.info(f"Inserted {total_inserted}/{len(records)} records")
            
            logger.info(f"Successfully inserted {total_inserted} records")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            self.conn.rollback()
            return False
    
    def verify_insertion(self):
        """Verify data was inserted correctly"""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("DATA VERIFICATION REPORT")
            logger.info("=" * 60)
            
            # Total count
            self.cursor.execute("SELECT COUNT(*) FROM airline_tweets")
            total_count = self.cursor.fetchone()[0]
            logger.info(f"Total records in database: {total_count}")
            
            # Count by sentiment
            self.cursor.execute("""
                SELECT airline_sentiment, COUNT(*) 
                FROM airline_tweets 
                GROUP BY airline_sentiment 
                ORDER BY airline_sentiment
            """)
            sentiment_counts = self.cursor.fetchall()
            logger.info("\nRecords by sentiment:")
            for sentiment, count in sentiment_counts:
                logger.info(f"  - {sentiment}: {count}")
            
            # Count by airline
            self.cursor.execute("""
                SELECT airline, COUNT(*) 
                FROM airline_tweets 
                GROUP BY airline 
                ORDER BY COUNT(*) DESC
            """)
            airline_counts = self.cursor.fetchall()
            logger.info("\nRecords by airline:")
            for airline, count in airline_counts:
                logger.info(f"  - {airline}: {count}")
            
            # Sample records
            self.cursor.execute("""
                SELECT id, airline_sentiment, airline, 
                       LEFT(text, 50) as text_preview, tweet_created 
                FROM airline_tweets 
                LIMIT 5
            """)
            samples = self.cursor.fetchall()
            logger.info("\nSample records:")
            for record in samples:
                logger.info(f"  ID: {record[0]}, Sentiment: {record[1]}, "
                          f"Airline: {record[2]}, Text: {record[3]}...")
            
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Error verifying data: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connections closed")


def main():
    """Main execution function"""
    # Get the data file path (relative to backend/database/)
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'processed', 'data.csv'
    )
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False
    
    logger.info(f"Loading data from: {data_path}")
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from CSV file")
        
        # Initialize database
        db = DatabaseInitializer()
        
        # Step 1: Connect to PostgreSQL
        if not db.connect_to_postgres():
            return False
        
        # Step 2: Create database
        if not db.create_database():
            return False
        
        # Step 3: Connect to the database
        if not db.connect_to_database():
            return False
        
        # Step 4: Create tables
        if not db.create_tables():
            return False
        
        # Step 5: Validate data
        is_valid, df_clean, validation_report = db.validate_data(df)
        
        if not is_valid:
            logger.error("Data validation failed")
            return False
        
        # Step 6: Insert data
        if not db.insert_data(df_clean):
            return False
        
        # Step 7: Verify insertion
        db.verify_insertion()
        
        # Close connections
        db.close()
        
        logger.info("\n✓ Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return False


if __name__ == "__main__":
    main()


