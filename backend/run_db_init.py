"""
Script to run database initialization
"""
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from database.init import main

if __name__ == "__main__":
    print("Starting database initialization...")
    print("Make sure PostgreSQL is running (docker-compose up postgres)")
    print("-" * 60)
    
    success = main()
        
    if success:
        print("\n✓ Database initialization completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Database initialization failed!")
        sys.exit(1)
