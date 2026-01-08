#!/usr/bin/env python
"""Quick validation script for Decision Intelligence System

Run with: python scripts/validate_all.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_data_files():
    """Check required data files exist"""
    print("1. Checking data files...")
    data_files = [
        ('data/processed/canonical_events.parquet', True),
        ('data/outputs/uplift_scores.parquet', True),
        ('data/outputs/refutation_report.json', True),
        ('data/outputs/data_quality_report.json', True),
        ('data/outputs/drift_report.json', False),  # Optional
    ]
    
    results = []
    for f, required in data_files:
        exists = os.path.exists(f)
        status = '✅' if exists else ('❌' if required else '⚠️')
        print(f"   {status} {f}")
        if required:
            results.append(exists)
    
    return all(results)


def check_model_files():
    """Check model files exist"""
    print("\n2. Checking model files...")
    model_files = [
        'models/causal_forest_v1.joblib',
    ]
    
    results = []
    for f in model_files:
        exists = os.path.exists(f)
        print(f"   {'✅' if exists else '❌'} {f}")
        results.append(exists)
    
    return all(results)


def check_imports():
    """Test all required imports"""
    print("\n3. Testing imports...")
    
    imports = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('sklearn', 'sklearn'),
        ('econml', 'econml'),
        ('dowhy', 'dowhy'),
        ('CausalEstimator', 'src.causal.estimation'),
        ('RefutationTester', 'src.causal.refutation'),
        ('ModelRegistry', 'src.services.model_registry'),
        ('DriftDetector', 'src.services.drift_detection'),
        ('NotificationService', 'src.services.notifications'),
    ]
    
    results = []
    for name, module in imports:
        try:
            if '.' in module:
                exec(f"from {module} import {name}")
            else:
                __import__(module)
            print(f"   ✅ {name}")
            results.append(True)
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            results.append(False)
    
    return all(results)


def check_model_registry():
    """Test Model Registry service"""
    print("\n4. Testing Model Registry...")
    try:
        from src.services.model_registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.list_models()
        prod = registry.get_production_model()
        
        print(f"   ✅ Registry initialized")
        print(f"   ✅ {len(models)} model(s) registered")
        if prod:
            print(f"   ✅ Production model: {prod.get('model_id', 'N/A')}")
        else:
            print(f"   ⚠️ No production model set")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_drift_detection():
    """Test Drift Detection service"""
    print("\n5. Testing Drift Detection...")
    try:
        from src.services.drift_detection import DriftDetector
        import pandas as pd
        import numpy as np
        
        # Create small test data
        test_data = pd.DataFrame({
            'age': np.random.normal(40, 10, 100),
            'tenure_months': np.random.exponential(12, 100),
            'pre_revenue': np.random.lognormal(5, 1, 100)
        })
        
        # Initialize detector with baseline data
        detector = DriftDetector(baseline_df=test_data)
        
        # Test baseline computation
        detector.compute_baseline_stats(test_data)
        print(f"   ✅ Baseline stats computed")
        
        # Test drift detection (no features argument - uses baseline features)
        drift_result = detector.detect_drift(test_data, alert_on_drift=False)
        print(f"   ✅ Drift detection works")
        print(f"   ✅ Drift detected: {drift_result.get('drift_detected', False)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_notifications():
    """Test Notification service"""
    print("\n6. Testing Notifications...")
    try:
        from src.services.notifications import NotificationService
        
        notifier = NotificationService()
        print(f"   ✅ Notification service initialized")
        
        # Check if webhook is configured
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if webhook_url:
            print(f"   ✅ Slack webhook configured")
        else:
            print(f"   ⚠️ Slack webhook not configured (SLACK_WEBHOOK_URL)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_causal_estimation():
    """Test Causal Estimation"""
    print("\n7. Testing Causal Estimation...")
    try:
        from src.causal.estimation import CausalEstimator
        import pandas as pd
        import numpy as np
        
        # Check if data exists for real test (use correct filename)
        if os.path.exists('data/processed/canonical_events.parquet'):
            df = pd.read_parquet('data/processed/canonical_events.parquet')
            print(f"   ✅ Loaded {len(df)} records")
            
            # Quick stats
            if 'treatment' in df.columns and 'outcome' in df.columns:
                ate_approx = df[df['treatment'] == 1]['outcome'].mean() - df[df['treatment'] == 0]['outcome'].mean()
                print(f"   ✅ Naive ATE estimate: ${ate_approx:.2f}")
            
            # CausalEstimator requires df in __init__
            estimator = CausalEstimator(df)
            print(f"   ✅ CausalEstimator initialized")
        else:
            print(f"   ⚠️ No data file - run pipeline first: python run_pipeline.py")
            # Create dummy data to test import
            dummy_df = pd.DataFrame({
                'user_id': range(100),
                'treatment': np.random.binomial(1, 0.5, 100),
                'outcome': np.random.normal(100, 20, 100),
                'age': np.random.normal(40, 10, 100)
            })
            estimator = CausalEstimator(dummy_df)
            print(f"   ✅ CausalEstimator initialized with dummy data")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_refutation():
    """Test Refutation report"""
    print("\n8. Testing Refutation Report...")
    try:
        import json
        
        report_path = 'data/outputs/refutation_report.json'
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            
            overall = report.get('overall_pass', False)
            tests = report.get('tests', {})
            summary = report.get('summary', {})
            
            print(f"   ✅ Report loaded")
            print(f"   ✅ Overall pass: {overall}")
            print(f"   ✅ Tests: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed")
            
            for test_name, test_data in tests.items():
                passed = test_data.get('passed', 'False')
                passed_bool = passed if isinstance(passed, bool) else str(passed).lower() == 'true'
                status = '✅' if passed_bool else '❌'
                print(f"      {status} {test_name}")
            
            return True
        else:
            print(f"   ⚠️ No refutation report found")
            return True  # Not a failure, just not generated yet
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_streamlit_app():
    """Check Streamlit app can be imported"""
    print("\n9. Testing Streamlit App...")
    try:
        # Check config imports that are actually used
        from src.utils.config import PROJECT_ROOT, FILE_PATHS, MODEL_CONFIG
        print(f"   ✅ Config module loads")
        
        # Check app file exists
        app_path = 'src/streamlit_app/app.py'
        if os.path.exists(app_path):
            print(f"   ✅ Streamlit app file exists")
            
            # Basic syntax check with proper encoding
            with open(app_path, encoding='utf-8') as f:
                code = f.read()
            compile(code, app_path, 'exec')
            print(f"   ✅ Streamlit app syntax OK")
        
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_api():
    """Check FastAPI app"""
    print("\n10. Testing FastAPI...")
    try:
        from src.api.main import app
        print(f"   ✅ FastAPI app imports")
        
        # Check routes
        routes = [r.path for r in app.routes]
        print(f"   ✅ Routes: {routes}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all validation checks"""
    print("="*60)
    print("Decision Intelligence System - Validation Script")
    print("="*60)
    
    checks = [
        ("Data Files", check_data_files),
        ("Model Files", check_model_files),
        ("Imports", check_imports),
        ("Model Registry", check_model_registry),
        ("Drift Detection", check_drift_detection),
        ("Notifications", check_notifications),
        ("Causal Estimation", check_causal_estimation),
        ("Refutation Report", check_refutation),
        ("Streamlit App", check_streamlit_app),
        ("FastAPI", check_api),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = '✅ PASS' if result else '❌ FAIL'
        print(f"  {status}  {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"Total: {passed}/{len(results)} checks passed")
    
    if failed == 0:
        print("\n🎉 All validations PASSED!")
        return 0
    else:
        print(f"\n⚠️ {failed} validation(s) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
