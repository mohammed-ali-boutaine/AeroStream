"""
Script to run database initialization
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from database.init import main # type:ignore

if __name__ == "__main__":
    
    success = main()
        
    if success:
        print("\ndatabase initialization completed successfully!")
        sys.exit(0)
    else:
        print("\ndatabase initialization failed!")
        sys.exit(1)
