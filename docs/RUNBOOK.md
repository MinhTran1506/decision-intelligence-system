# Decision Intelligence Studio - Operational Runbook

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Weekly Operations](#weekly-operations)
3. [Monthly Operations](#monthly-operations)
4. [Incident Response](#incident-response)
5. [Rollback Procedures](#rollback-procedures)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Monitoring & Alerts](#monitoring--alerts)

---

## Daily Operations

### Morning Checklist (9:00 AM)

1. **Verify Pipeline Success**
   ```bash
   # Check Airflow DAG status
   airflow dags list-runs -d decision_intelligence_pipeline --state success -o table
   
   # Or check local logs
   tail -100 logs/pipeline_*.log | grep -E "(SUCCESS|FAILED|ERROR)"
   ```

2. **Review Key Metrics**
   - Open Streamlit dashboard: http://localhost:8501
   - Check Overview tab for:
     - ATE estimate (should be stable ±10%)
     - Refutation pass rate (should be ≥80%)
     - Total customers scored

3. **Check Drift Alerts**
   ```bash
   cat data/outputs/drift_report.json | python -m json.tool
   ```
   - If `drift_detected: true`, investigate features listed

4. **Review Active A/B Tests**
   - Navigate to A/B Test Tracking tab
   - Check running tests for sufficient sample sizes
   - Note any tests approaching statistical significance

### Evening Wrap-up (5:00 PM)

1. **Verify Data Freshness**
   ```bash
   # Check file modification times
   ls -la data/outputs/
   ```

2. **Review Any Errors**
   ```bash
   grep -i "error\|exception\|failed" logs/*.log | tail -20
   ```

---

## Weekly Operations

### Monday - Full Pipeline Review

1. **Run Full Pipeline with Refutation**
   ```bash
   python run_pipeline.py  # Full mode, includes all refutation tests
   ```

2. **Review Model Calibration**
   - Compare predicted vs observed uplift from completed A/B tests
   - Update calibration chart in Historical Results tab

3. **Check Model Registry**
   ```python
   from src.services.model_registry import get_registry
   registry = get_registry()
   models = registry.list_models(limit=10)
   for m in models:
       print(f"{m['model_id']}: {m['status']} - ATE=${m['ate_estimate']:.2f}")
   ```

### Wednesday - Data Quality Deep Dive

1. **Review Data Quality Report**
   ```bash
   cat data/outputs/data_quality_report.json | python -m json.tool
   ```

2. **Check Covariate Balance**
   - Ensure treatment/control groups are balanced
   - Flag any propensity score issues

### Friday - Performance & Drift Report

1. **Generate Weekly Summary**
   ```python
   from src.services.drift_detection import run_drift_detection
   from src.services.model_registry import get_registry
   
   # Run drift detection
   drift = run_drift_detection()
   
   # Get production model metrics
   registry = get_registry()
   prod_model = registry.get_production_model()
   
   print(f"Production Model: {prod_model['model_id']}")
   print(f"ATE: ${prod_model['ate_estimate']:.2f}")
   print(f"Drift Detected: {drift.get('drift_detected', False)}")
   ```

2. **Update Baseline if Stable**
   ```python
   from src.services.drift_detection import update_baseline
   update_baseline()  # Only if no significant drift
   ```

---

## Monthly Operations

### First Monday - Full Retrain

1. **Run Complete Retraining**
   ```bash
   # Clear cached data
   rm -rf data/processed/*
   rm -rf data/outputs/*
   
   # Run full pipeline
   python run_pipeline.py
   ```

2. **Review Causal Graph**
   - Validate adjustment set is still appropriate
   - Consider new confounders from business changes

3. **Model Comparison**
   - Compare new model vs previous production model
   - Only promote if metrics improve or remain stable

### Second Week - Stakeholder Report

1. **Generate Executive Summary**
   - ATE trend over past month
   - A/B test validation results
   - ROI from targeted campaigns
   - Recommendations for next month

2. **Calibration Analysis**
   - Plot predicted vs observed uplift
   - Calculate MAPE (Mean Absolute Percentage Error)
   - Flag if calibration degraded >20%

---

## Incident Response

### Pipeline Failure

**Severity: HIGH**

1. **Immediate Actions**
   ```bash
   # Check last error
   tail -50 logs/pipeline_*.log | grep -A5 "ERROR"
   
   # Identify failed step
   airflow tasks states-for-dag-run decision_intelligence_pipeline <run_id>
   ```

2. **Common Failures & Fixes**

   | Error | Cause | Fix |
   |-------|-------|-----|
   | `Missing required columns` | Schema change | Update canonical schema in config.py |
   | `Null rate > threshold` | Data quality issue | Check source data, impute or filter |
   | `No propensity overlap` | Selection bias | Review treatment assignment logic |
   | `Refutation failed` | Model issues | Check for omitted confounders |

3. **Escalation**
   - If not resolved in 30 min, page on-call engineer
   - If data issue, contact data platform team

### Drift Alert

**Severity: MEDIUM**

1. **Investigate Drift Source**
   ```python
   import json
   with open('data/outputs/drift_report.json') as f:
       report = json.load(f)
   
   for feat in report['features_with_drift']:
       info = report['feature_results'][feat]
       print(f"{feat}: KS={info['ks_statistic']:.4f}, "
             f"shift={info.get('mean_shift', 'N/A')}")
   ```

2. **Determine Action**
   - **Minor drift (KS < 0.2)**: Monitor, no action
   - **Moderate drift (0.2 ≤ KS < 0.3)**: Schedule retrain
   - **Severe drift (KS ≥ 0.3)**: Immediate retrain, notify stakeholders

3. **If Retrain Needed**
   ```bash
   # Update baseline with new data
   python -c "from src.services.drift_detection import update_baseline; update_baseline()"
   
   # Run pipeline
   python run_pipeline.py
   ```

### Model Performance Degradation

**Severity: HIGH**

1. **Detect via A/B Test**
   - Predicted uplift vs observed differs by >30%
   - Confidence intervals don't overlap

2. **Response**
   ```bash
   # Rollback to previous model
   python -c "
   from src.services.model_registry import get_registry
   registry = get_registry()
   registry.rollback()
   print('Rolled back to previous model')
   "
   ```

3. **Root Cause Analysis**
   - Check for data leakage
   - Review feature engineering
   - Validate causal assumptions

---

## Rollback Procedures

### Rollback Model

```python
from src.services.model_registry import get_registry

registry = get_registry()

# Option 1: Rollback to previous version
registry.rollback()

# Option 2: Rollback to specific version
registry.rollback(to_model_id="causal_forest_marketing_v20240115_093022")

# Verify
prod = registry.get_production_model()
print(f"Active model: {prod['model_id']}")
```

### Rollback Pipeline

```bash
# Restore previous outputs
cd data/outputs
git checkout HEAD~1 -- uplift_scores.parquet
git checkout HEAD~1 -- refutation_report.json

# Restore previous model
cd models
git checkout HEAD~1 -- causal_forest_v1.joblib
```

### Emergency: Disable Recommendations

```python
# Set all uplift scores to 0 (disable treatment)
import pandas as pd
from src.utils.config import FILE_PATHS

df = pd.read_parquet(FILE_PATHS["uplift_scores"])
df['uplift_score'] = 0
df['segment_name'] = 'Do Not Treat'
df.to_parquet(FILE_PATHS["uplift_scores"])

print("⚠️ All recommendations disabled")
```

---

## Troubleshooting Guide

### Uplift Scores Are Unstable

**Symptoms**: Large variance in uplift estimates between runs

**Diagnosis**:
```python
import pandas as pd
from src.utils.config import FILE_PATHS

df = pd.read_parquet(FILE_PATHS["uplift_scores"])
print(f"Uplift std: {df['uplift_score'].std():.2f}")
print(f"Uplift range: {df['uplift_score'].min():.2f} to {df['uplift_score'].max():.2f}")

# Check CI width
if 'uplift_ci_lower' in df.columns:
    ci_width = df['uplift_ci_upper'] - df['uplift_ci_lower']
    print(f"Avg CI width: {ci_width.mean():.2f}")
```

**Solutions**:
- Increase `n_estimators` in model config
- Increase `min_samples_leaf` for regularization
- Check for propensity score overlap issues

### Refutation Tests Failing

**Symptoms**: One or more refutation tests fail

**Diagnosis**:
```bash
cat data/outputs/refutation_report.json | python -m json.tool | grep -A5 '"passed": false'
```

**Solutions by Test**:

| Test | Failure Meaning | Action |
|------|-----------------|--------|
| Placebo | Spurious correlation | Review causal graph, add confounders |
| Random Cause | Sensitive to noise | Increase sample size |
| Subset | Unstable estimates | Check for heterogeneity |
| Bootstrap | Wide CIs | Need more data or simpler model |

### API Not Responding

**Symptoms**: 503 or connection refused

**Diagnosis**:
```bash
# Check if running
curl http://localhost:8000/health
curl http://localhost:8001/health

# Check logs
tail -50 logs/api_*.log
```

**Solutions**:
```bash
# Restart API
pkill -f "uvicorn"
cd src/api && uvicorn main:app --host 0.0.0.0 --port 8000 &

# Check port conflicts
netstat -tulpn | grep 8000
```

---

## Monitoring & Alerts

### Key Metrics to Monitor

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| ATE Estimate | $40-50 | ±20% change |
| Refutation Pass Rate | ≥80% | <60% |
| Drift KS Statistic | <0.15 | >0.20 |
| Pipeline Duration | 5-10 min | >30 min |
| API Latency (p95) | <500ms | >2000ms |

### Setting Up Alerts

1. **Slack Notifications**
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
   ```

2. **Prometheus Metrics** (if deployed)
   ```yaml
   # prometheus.yml
   - job_name: 'decision-intel'
     static_configs:
       - targets: ['localhost:8000']
     metrics_path: '/metrics'
   ```

3. **Grafana Dashboard**
   - Import dashboard from `monitoring/grafana/dashboard.json`
   - Key panels: ATE trend, refutation pass rate, drift alerts

---

## Contact Information

| Role | Contact | When to Page |
|------|---------|--------------|
| On-Call Engineer | oncall@company.com | Pipeline failures, API down |
| Data Platform | data-platform@company.com | Source data issues |
| ML Engineering | ml-eng@company.com | Model degradation |
| Business Stakeholder | marketing-analytics@company.com | Campaign impact questions |

---

## Appendix: Useful Commands

```bash
# Quick health check
python test_installation.py

# Run pipeline in quick mode
python run_pipeline.py --quick

# Check model registry
python -c "from src.services.model_registry import get_registry; r=get_registry(); print(r.list_models())"

# Run drift detection
python -c "from src.services.drift_detection import run_drift_detection; run_drift_detection()"

# Start Streamlit
streamlit run src/streamlit_app/app.py

# Start API
uvicorn src.api.main:app --reload --port 8000
```
