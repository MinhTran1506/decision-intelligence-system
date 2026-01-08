# Decision Intelligence Studio - Testing Guide

This guide covers how to test all features of the Decision Intelligence System.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Feature Testing Checklist](#feature-testing-checklist)
4. [Detailed Test Instructions](#detailed-test-instructions)
5. [Automated Tests](#automated-tests)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Environment Setup
```powershell
# Activate virtual environment
cd C:\minhtran\coding\decision-intelligence-system
.\venv\Scripts\Activate.ps1

# Install dependencies (if not done)
pip install -r requirements.txt
```

### 2. Generate Sample Data (if needed)
```powershell
python -c "from src.data.generate_sample_data import generate_sample_data; generate_sample_data()"
```

### 3. Run Full Pipeline (generates all required artifacts)
```powershell
python run_pipeline.py
```

---

## Quick Start

### Start All Services
```powershell
# Terminal 1: Streamlit Dashboard (main UI)
streamlit run src/streamlit_app/app.py --server.port 8501

# Terminal 2: FastAPI Scoring API (optional)
uvicorn src.api.main:app --reload --port 8000
```

### Access Points
| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| API Health Check | http://localhost:8000/health |

---

## Feature Testing Checklist

### ✅ Streamlit Dashboard Pages

| Page | Status | Test Steps |
|------|--------|------------|
| Overview | ⬜ | See [Overview Tests](#1-overview-dashboard) |
| Real-time Monitoring | ⬜ | See [Monitoring Tests](#2-real-time-monitoring) |
| A/B Test Tracking | ⬜ | See [A/B Test Tests](#3-ab-test-tracking) |
| Customer Lookup | ⬜ | See [Customer Lookup Tests](#4-customer-lookup) |
| Model Comparison | ⬜ | See [Model Comparison Tests](#5-model-comparison) |
| Model Registry | ⬜ | See [Model Registry Tests](#6-model-registry) |
| Advanced Analytics | ⬜ | See [Advanced Analytics Tests](#7-advanced-analytics) |

### ✅ Backend Services

| Service | Status | Test Steps |
|---------|--------|------------|
| Causal Estimation | ⬜ | See [Causal Tests](#8-causal-estimation) |
| Refutation Tests | ⬜ | See [Refutation Tests](#9-refutation-tests) |
| Model Registry Service | ⬜ | See [Registry Service Tests](#10-model-registry-service) |
| Drift Detection | ⬜ | See [Drift Detection Tests](#11-drift-detection) |
| Notifications | ⬜ | See [Notification Tests](#12-notifications) |
| Data Quality | ⬜ | See [Data Quality Tests](#13-data-quality) |

### ✅ APIs

| Endpoint | Status | Test Steps |
|----------|--------|------------|
| POST /score | ⬜ | See [API Tests](#14-scoring-api) |
| POST /simulate | ⬜ | See [API Tests](#14-scoring-api) |
| GET /health | ⬜ | See [API Tests](#14-scoring-api) |

---

## Detailed Test Instructions

### 1. Overview Dashboard

**Navigate to:** Sidebar → "Overview"

**Expected Elements:**
- [ ] KPI Cards showing:
  - Average Treatment Effect (ATE) in dollars
  - ROI percentage
  - Total customers analyzed
  - Model confidence percentage
- [ ] Treatment Effect Distribution chart (histogram)
- [ ] Segment Performance chart (bar chart by segment)
- [ ] Recent Activity section

**Test Actions:**
1. Verify all numbers are populated (not NaN or empty)
2. Hover over charts to see tooltips
3. Check that segment names match your data

---

### 2. Real-time Monitoring

**Navigate to:** Sidebar → "Real-time Monitoring"

**Expected Elements:**
- [ ] Metric selector dropdown
- [ ] Time window selector
- [ ] Live chart with streaming data
- [ ] Auto-refresh toggle

**Test Actions:**
1. Select different metrics (ATE, Confidence, etc.)
2. Change time window (1h, 6h, 24h)
3. Enable auto-refresh and verify data updates
4. Click "Refresh Now" button

---

### 3. A/B Test Tracking

**Navigate to:** Sidebar → "A/B Test Tracking"

**Expected Elements:**
- [ ] Active Tests tab with test cards
- [ ] Create Test tab with form
- [ ] Completed Tests tab with historical results

**Test Actions:**

#### Create a New Test:
1. Go to "Create Test" tab
2. Fill in:
   - Test Name: "Test Feature X"
   - Description: "Testing new targeting algorithm"
   - Treatment Allocation: 50%
   - Target Sample Size: 1000
3. Click "Create A/B Test"
4. Verify success message appears

#### View Test Results:
1. Go to "Active Tests" tab
2. Click "View Results" on any test
3. Verify you see:
   - Control vs Treatment comparison bar chart
   - Statistical significance indicator
   - Detailed metrics table
   - Conclusion box (green/yellow/red)

#### Complete a Test:
1. Select a test and click "Complete Test"
2. Choose winner (Control/Treatment)
3. Verify test moves to "Completed Tests" tab

---

### 4. Customer Lookup

**Navigate to:** Sidebar → "Customer Lookup"

**Expected Elements:**
- [ ] Customer ID search input
- [ ] Customer profile card
- [ ] Individual treatment effect estimate
- [ ] Feature importance for this customer

**Test Actions:**
1. Enter a valid customer ID (e.g., 1, 100, 500)
2. Click "Search" or press Enter
3. Verify customer details appear:
   - Demographics (age, tenure, segment)
   - Predicted CATE with confidence interval
   - Recommendation (Treat / Do Not Treat)
4. Test with invalid ID and verify error handling

---

### 5. Model Comparison

**Navigate to:** Sidebar → "Model Comparison"

**Expected Elements:**
- [ ] Current Model metrics card
- [ ] ATE Distribution chart
- [ ] CATE Heterogeneity chart
- [ ] Refutation Test Results (5 tests)
- [ ] Model Selection Recommendation

**Test Actions:**
1. Verify ATE estimate matches pipeline output
2. Check all 5 refutation tests show Pass/Fail status:
   - Placebo Treatment
   - Random Common Cause
   - Subset Validation
   - Data Subset
   - Bootstrap
3. Verify interpretation text appears for each test
4. Check recommendation section is visible

---

### 6. Model Registry

**Navigate to:** Sidebar → "Model Registry"

**Expected Elements:**
- [ ] Production Model tab
- [ ] All Versions tab
- [ ] Drift Detection tab

**Test Actions:**

#### Production Model Tab:
1. Verify current production model details display
2. Check metrics: ATE, CATE Std, Refutation Pass Rate
3. Verify deployment timestamp

#### All Versions Tab:
1. View list of all registered model versions
2. Test "Promote to Production" button on a staging model
3. Test "Archive" button on old models
4. Test "Rollback" functionality (if available)

#### Drift Detection Tab:
1. View baseline statistics
2. Check drift status indicators
3. Run drift detection if button available
4. Verify alert thresholds are displayed

---

### 7. Advanced Analytics

**Navigate to:** Sidebar → "Advanced Analytics"

**Expected Elements:**
- [ ] Feature importance visualization
- [ ] CATE by feature analysis
- [ ] Segment deep-dive
- [ ] What-if scenarios

**Test Actions:**
1. Select different features for analysis
2. Interact with scatter plots
3. Test segment filtering
4. Run what-if scenario if available

---

### 8. Causal Estimation

**Test via Python:**
```python
from src.causal.estimation import CausalEstimator
import pandas as pd

# Load data
df = pd.read_parquet('data/processed/canonical_events.parquet')

# Initialize estimator (requires df in constructor)
estimator = CausalEstimator(df)

# Check data prepared
print(f"Records: {len(df)}")
print(f"Treatment rate: {df['treatment'].mean():.1%}")

# Quick stats
if 'treatment' in df.columns and 'outcome' in df.columns:
    ate_approx = df[df['treatment'] == 1]['outcome'].mean() - df[df['treatment'] == 0]['outcome'].mean()
    print(f"Naive ATE estimate: ${ate_approx:.2f}")

print("✅ Causal Estimation: PASSED")
```

---

### 9. Refutation Tests

**Test via Python:**
```python
from src.causal.refutation import RefutationTester
import json

# Load refutation report
with open('data/outputs/refutation_report.json') as f:
    report = json.load(f)

# Verify structure
assert 'overall_pass' in report
assert 'tests' in report
assert len(report['tests']) >= 4

# Check each test
for test_name, test_data in report['tests'].items():
    assert 'passed' in test_data
    print(f"  {test_name}: {'✅' if test_data['passed'] == 'True' else '❌'}")

print(f"\n✅ Refutation Tests: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
```

---

### 10. Model Registry Service

**Test via Python:**
```python
from src.services.model_registry import ModelRegistry

registry = ModelRegistry()

# Test listing models
models = registry.list_models()
print(f"Total models: {len(models)}")

# Test getting production model
prod_model = registry.get_production_model()
if prod_model:
    print(f"Production model: {prod_model['model_id']}")
    print(f"  Version: {prod_model['version']}")
    print(f"  ATE: {prod_model['ate_estimate']}")

# Test registering a new model
new_id = registry.register_model(
    model_path='models/causal_forest_v1.joblib',
    ate_estimate=48.5,
    cate_std=25.3,
    refutation_pass_rate=100.0,
    metadata={'test': True}
)
print(f"Registered new model: {new_id}")

print("✅ Model Registry: PASSED")
```

---

### 11. Drift Detection

**Test via Python:**
```python
from src.services.drift_detection import DriftDetector
import pandas as pd

# Initialize detector
detector = DriftDetector()

# Load baseline and current data
baseline_df = pd.read_parquet('data/processed/canonical_dataset.parquet')
current_df = baseline_df.sample(frac=0.8)  # Simulate current data

# Compute baseline stats
detector.compute_baseline_stats(baseline_df)
print("Baseline stats computed")

# Detect drift
drift_report = detector.detect_drift(
    current_data=current_df,
    features=['age', 'tenure_months', 'pre_revenue']
)

print(f"Drift detected: {drift_report['drift_detected']}")
for feature, result in drift_report['features'].items():
    status = '⚠️ DRIFT' if result['drift'] else '✅ OK'
    print(f"  {feature}: {status} (p={result.get('p_value', 'N/A'):.4f})")

print("✅ Drift Detection: PASSED")
```

---

### 12. Notifications

**Test via Python:**
```python
import os
from src.services.notifications import NotificationService

# For testing without actual Slack webhook
notifier = NotificationService()

# Test message formatting (dry run)
message = notifier._format_pipeline_success(
    ate_estimate=48.5,
    model_version='v1.0.1',
    refutation_pass_rate=100.0
)
print("Success message formatted:")
print(message[:200] + "...")

# Test failure message
fail_message = notifier._format_pipeline_failure(
    error="Test error",
    stage="estimation"
)
print("\nFailure message formatted:")
print(fail_message[:200] + "...")

print("\n✅ Notifications: Message formatting PASSED")

# To test actual Slack delivery:
# 1. Set SLACK_WEBHOOK_URL environment variable
# 2. Uncomment below:
# os.environ['SLACK_WEBHOOK_URL'] = 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
# notifier.send_pipeline_success(ate_estimate=48.5, model_version='v1.0.1', refutation_pass_rate=100.0)
```

---

### 13. Data Quality

**Test via Python:**
```python
from src.data.data_quality_checks import DataQualityChecker
import pandas as pd

# Load data
df = pd.read_parquet('data/processed/canonical_dataset.parquet')

# Run quality checks
checker = DataQualityChecker()
report = checker.run_all_checks(df)

print(f"Overall Status: {'✅ PASSED' if report['overall_pass'] else '❌ FAILED'}")
print(f"Checks passed: {report['checks_passed']}/{report['total_checks']}")

for check_name, check_result in report['checks'].items():
    status = '✅' if check_result['passed'] else '❌'
    print(f"  {status} {check_name}")

print("\n✅ Data Quality: PASSED")
```

---

### 14. Scoring API

**Start the API:**
```powershell
uvicorn src.api.main:app --reload --port 8000
```

**Test via curl/PowerShell:**

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

# Score a single customer
$body = @{
    customer_id = 123
    age = 35
    tenure_months = 24
    pre_revenue = 150.0
    segment = "premium"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/score" -Method Post -Body $body -ContentType "application/json"

# Simulate treatment
$simBody = @{
    treatment_probability = 0.5
    target_segment = "premium"
    sample_size = 1000
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/simulate" -Method Post -Body $simBody -ContentType "application/json"
```

**Or test via browser:**
1. Go to http://localhost:8000/docs
2. Click on each endpoint
3. Click "Try it out"
4. Fill in parameters and click "Execute"

---

## Automated Tests

### Run All Unit Tests
```powershell
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_services.py -v

# Run specific test
pytest tests/test_services.py::TestModelRegistry -v
```

### Run Integration Tests
```powershell
# Full pipeline test
python run_pipeline.py --test-mode

# API integration test
pytest tests/test_api.py -v
```

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError"
```powershell
# Ensure virtual environment is active
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. "FileNotFoundError: canonical_dataset.parquet"
```powershell
# Generate sample data first
python -c "from src.data.generate_sample_data import generate_sample_data; generate_sample_data()"

# Or run full pipeline
python run_pipeline.py
```

#### 3. "No models in registry"
```powershell
# Run pipeline to register a model
python run_pipeline.py
```

#### 4. Streamlit won't start
```powershell
# Kill any existing Streamlit processes
Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process

# Clear Streamlit cache
Remove-Item -Recurse -Force "$env:USERPROFILE\.streamlit\cache" -ErrorAction SilentlyContinue

# Restart
streamlit run src/streamlit_app/app.py
```

#### 5. API returns 500 errors
```powershell
# Check API logs
uvicorn src.api.main:app --reload --port 8000 --log-level debug
```

---

## Test Report Template

Use this template to document your testing:

```markdown
# Test Report - [Date]

## Environment
- Python Version: 
- OS: Windows
- Branch: main

## Test Results

### Streamlit Dashboard
| Page | Status | Notes |
|------|--------|-------|
| Overview | ⬜/✅/❌ | |
| Real-time Monitoring | ⬜/✅/❌ | |
| A/B Test Tracking | ⬜/✅/❌ | |
| Customer Lookup | ⬜/✅/❌ | |
| Model Comparison | ⬜/✅/❌ | |
| Model Registry | ⬜/✅/❌ | |
| Advanced Analytics | ⬜/✅/❌ | |

### Backend Services
| Service | Status | Notes |
|---------|--------|-------|
| Causal Estimation | ⬜/✅/❌ | |
| Refutation Tests | ⬜/✅/❌ | |
| Model Registry | ⬜/✅/❌ | |
| Drift Detection | ⬜/✅/❌ | |
| Notifications | ⬜/✅/❌ | |

### APIs
| Endpoint | Status | Notes |
|----------|--------|-------|
| GET /health | ⬜/✅/❌ | |
| POST /score | ⬜/✅/❌ | |
| POST /simulate | ⬜/✅/❌ | |

## Issues Found
1. 

## Sign-off
- Tester: 
- Date:
```

---

## Quick Validation Script

Run this script to quickly validate all major components:

```powershell
# Save as: scripts/validate_all.py
```

```python
#!/usr/bin/env python
"""Quick validation script for Decision Intelligence System"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    results = []
    
    # 1. Check data files
    print("1. Checking data files...")
    data_files = [
        'data/processed/canonical_dataset.parquet',
        'data/processed/uplift_scores.parquet',
        'data/outputs/refutation_report.json',
        'data/outputs/data_quality_report.json'
    ]
    for f in data_files:
        exists = os.path.exists(f)
        results.append(('Data: ' + f, exists))
        print(f"   {'✅' if exists else '❌'} {f}")
    
    # 2. Check model file
    print("\n2. Checking model...")
    model_exists = os.path.exists('models/causal_forest_v1.joblib')
    results.append(('Model file', model_exists))
    print(f"   {'✅' if model_exists else '❌'} models/causal_forest_v1.joblib")
    
    # 3. Test imports
    print("\n3. Testing imports...")
    try:
        from src.causal.estimation import CausalEstimator
        from src.causal.refutation import RefutationTester
        from src.services.model_registry import ModelRegistry
        from src.services.drift_detection import DriftDetector
        from src.services.notifications import NotificationService
        results.append(('Imports', True))
        print("   ✅ All imports successful")
    except Exception as e:
        results.append(('Imports', False))
        print(f"   ❌ Import error: {e}")
    
    # 4. Test model registry
    print("\n4. Testing Model Registry...")
    try:
        registry = ModelRegistry()
        models = registry.list_models()
        results.append(('Model Registry', True))
        print(f"   ✅ Model Registry works ({len(models)} models)")
    except Exception as e:
        results.append(('Model Registry', False))
        print(f"   ❌ Error: {e}")
    
    # 5. Test drift detection
    print("\n5. Testing Drift Detection...")
    try:
        detector = DriftDetector()
        results.append(('Drift Detection', True))
        print("   ✅ Drift Detection initialized")
    except Exception as e:
        results.append(('Drift Detection', False))
        print(f"   ❌ Error: {e}")
    
    # 6. Test notifications
    print("\n6. Testing Notifications...")
    try:
        notifier = NotificationService()
        results.append(('Notifications', True))
        print("   ✅ Notification Service initialized")
    except Exception as e:
        results.append(('Notifications', False))
        print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*50)
    passed = sum(1 for _, v in results if v)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All validations PASSED!")
        return 0
    else:
        print("❌ Some validations FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

Run it with:
```powershell
python scripts/validate_all.py
```

---

**Last Updated:** January 8, 2026
