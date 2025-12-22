"""Database connection and utilities for backend."""
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'postgres_backend'),
    'database': os.getenv('DB_NAME', 'backend_db'),
    'user': os.getenv('DB_USER', 'ali'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'port': int(os.getenv('DB_PORT', 5432))
}

# Connection pool (initialized on first use)
_connection_pool = None


def get_connection_pool():
    """Get or create connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **DB_CONFIG
        )
    return _connection_pool


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def test_connection():
    """Test database connection and return table info."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM airline_tweets")
                count = cursor.fetchone()[0]
                return True, count
    except Exception as e:
        return False, str(e)
