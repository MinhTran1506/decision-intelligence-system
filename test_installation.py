#!/usr/bin/env python3
"""Test if all dependencies are installed correctly"""

def test_imports():
    """Test critical imports"""
    print("Testing imports...")
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import dowhy
        print("✓ dowhy")
    except ImportError as e:
        print(f"✗ dowhy: {e}")
        return False
    
    try:
        import econml
        print("✓ econml")
    except ImportError as e:
        print(f"✗ econml: {e}")
        return False
    
    try:
        import fastapi
        print("✓ fastapi")
    except ImportError as e:
        print(f"✗ fastapi: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ uvicorn")
    except ImportError as e:
        print(f"✗ uvicorn: {e}")
        return False
    
    print("\n✓ All critical imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
