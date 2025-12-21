import psycopg2
from psycopg2 import sql, Error
from typing import List, Dict, Any, Optional
import logging

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'root',
    'database': 'aerostream'
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_connection():
    """
    Create and return a database connection.
    
    Returns:
        connection: PostgreSQL database connection
    """
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        logger.info("Database connection established successfully")
        return connection
    except Error as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")
        raise


def close_connection(connection, cursor=None):
    """
    Close database connection and cursor.
    
    Args:
        connection: Database connection to close
        cursor: Database cursor to close (optional)
    """
    try:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logger.info("Database connection closed")
    except Error as e:
        logger.error(f"Error closing database connection: {e}")


def insert_data(table_name: str, data: Dict[str, Any]) -> bool:
    """
    Insert a single row of data into the specified table.
    
    Args:
        table_name: Name of the table to insert data into
        data: Dictionary with column names as keys and values to insert
        
    Returns:
        bool: True if insertion was successful, False otherwise
        
    Example:
        insert_data('users', {'name': 'John', 'email': 'john@example.com'})
    """
    connection = None
    cursor = None
    
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        # Build the INSERT query dynamically
        columns = data.keys()
        values = [data[column] for column in columns]
        
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(values))
        )
        
        cursor.execute(insert_query, values)
        connection.commit()
        
        logger.info(f"Data inserted successfully into {table_name}")
        return True
        
    except Error as e:
        logger.error(f"Error inserting data into {table_name}: {e}")
        if connection:
            connection.rollback()
        return False
        
    finally:
        close_connection(connection, cursor)


def insert_many(table_name: str, data_list: List[Dict[str, Any]]) -> bool:
    """
    Insert multiple rows of data into the specified table.
    
    Args:
        table_name: Name of the table to insert data into
        data_list: List of dictionaries, each containing column names and values
        
    Returns:
        bool: True if insertion was successful, False otherwise
        
    """
    connection = None
    cursor = None
    
    try:
        if not data_list:
            logger.warning("No data provided for insertion")
            return False
            
        connection = get_connection()
        cursor = connection.cursor()
        
        # Assume all dictionaries have the same keys
        columns = list(data_list[0].keys())
        
        # Build the INSERT query
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns))
        )
        
        # Prepare data for executemany
        values_list = [[row[col] for col in columns] for row in data_list]
        
        cursor.executemany(insert_query, values_list)
        connection.commit()
        
        logger.info(f"{len(data_list)} rows inserted successfully into {table_name}")
        return True
        
    except Error as e:
        logger.error(f"Error inserting multiple rows into {table_name}: {e}")
        if connection:
            connection.rollback()
        return False
        
    finally:
        close_connection(connection, cursor)


def execute_query(query: str, params: Optional[tuple] = None) -> Optional[List[tuple]]:
    """
    Execute a custom SQL query and return results.
    
    Args:
        query: SQL query to execute
        params: Optional parameters for the query
        
    Returns:
        List of tuples containing query results, or None if error
    """
    connection = None
    cursor = None
    
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        # For SELECT queries, fetch results
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            return results
        else:
            connection.commit()
            return None
            
    except Error as e:
        logger.error(f"Error executing query: {e}")
        if connection:
            connection.rollback()
        return None
        
    finally:
        close_connection(connection, cursor)