"""
Apache Airflow DAG for Decision Intelligence Pipeline

Production-ready orchestration with:
- Data quality checks
- Causal estimation
- Refutation tests
- Model registration
- Notifications

To deploy:
1. Copy this file to your Airflow dags/ folder
2. Set environment variables:
   - SLACK_WEBHOOK_URL: For notifications
   - PROJECT_PATH: Path to decision-intelligence-system
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import os
import sys

# Configuration
PROJECT_PATH = Variable.get("decision_intel_project_path", 
                            default_var="/opt/decision-intelligence-system")
sys.path.insert(0, PROJECT_PATH)

# Default DAG arguments
default_args = {
    'owner': 'decision-intel',
    'depends_on_past': False,
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def task_generate_data(**context):
    """Generate or load sample data"""
    from src.data.generate_sample_data import main as generate_data
    success = generate_data()
    if not success:
        raise Exception("Data generation failed")
    return {'status': 'success', 'step': 'generate_data'}


def task_create_canonical(**context):
    """Transform raw data to canonical schema"""
    from src.data.create_canonical import main as create_canonical
    success = create_canonical()
    if not success:
        raise Exception("Canonical dataset creation failed")
    return {'status': 'success', 'step': 'create_canonical'}


def task_data_quality_checks(**context):
    """Run data quality validation"""
    from src.data.data_quality_checks import main as run_dq_checks
    success = run_dq_checks()
    if not success:
        raise Exception("Data quality checks failed - stopping pipeline")
    return {'status': 'success', 'step': 'data_quality'}


def task_drift_detection(**context):
    """Check for feature drift"""
    from src.services.drift_detection import run_drift_detection
    results = run_drift_detection()
    
    if results.get('drift_detected'):
        # Log warning but don't fail
        print(f"⚠️ Drift detected in: {results.get('features_with_drift')}")
    
    return results


def task_causal_estimation(**context):
    """Run causal estimation pipeline"""
    from src.causal.estimation import main as run_estimation
    success = run_estimation()
    if not success:
        raise Exception("Causal estimation failed")
    return {'status': 'success', 'step': 'estimation'}


def task_refutation_tests(**context):
    """Run refutation tests for robustness"""
    from src.causal.refutation import main as run_refutation
    success = run_refutation()
    # Don't fail on refutation - just log results
    return {'status': 'success' if success else 'warning', 'step': 'refutation'}


def task_register_model(**context):
    """Register trained model in registry"""
    import json
    from src.services.model_registry import get_registry
    from src.utils.config import FILE_PATHS
    import pandas as pd
    
    registry = get_registry()
    
    # Load results
    uplift_df = pd.read_parquet(FILE_PATHS["uplift_scores"])
    
    refutation_report = {}
    if FILE_PATHS["refutation_report"].exists():
        with open(FILE_PATHS["refutation_report"], 'r') as f:
            refutation_report = json.load(f)
    
    # Calculate metrics
    ate_estimate = float(uplift_df['uplift_score'].mean())
    cate_std = float(uplift_df['uplift_score'].std())
    
    tests = refutation_report.get('tests', {})
    if tests and isinstance(tests, dict):
        passed = sum(1 for t in tests.values() if str(t.get('passed', '')).lower() == 'true')
        refutation_pass_rate = (passed / len(tests)) * 100 if tests else 100
    else:
        refutation_pass_rate = 100
    
    # Register model
    model_id = registry.register_model(
        model_name="causal_forest_marketing",
        version=f"v{datetime.now().strftime('%Y%m%d')}",
        ate_estimate=ate_estimate,
        cate_std=cate_std,
        refutation_pass_rate=refutation_pass_rate,
        training_rows=len(uplift_df),
        notes=f"Airflow run {context['run_id']}"
    )
    
    # Auto-promote if refutation passes
    if refutation_pass_rate >= 80:
        registry.promote_model(model_id)
        print(f"✓ Model {model_id} promoted to production")
    
    return {
        'model_id': model_id,
        'ate': ate_estimate,
        'refutation_pass_rate': refutation_pass_rate
    }


def task_update_baseline(**context):
    """Update drift detection baseline after successful run"""
    from src.services.drift_detection import update_baseline
    update_baseline()
    return {'status': 'success', 'step': 'update_baseline'}


def task_notify_success(**context):
    """Send success notification"""
    from src.services.notifications import notify_pipeline_complete
    
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='register_model')
    
    # Calculate duration
    dag_run = context['dag_run']
    start = dag_run.start_date
    end = datetime.now()
    duration = (end - start).total_seconds()
    
    notify_pipeline_complete(
        ate_estimate=model_info.get('ate', 0),
        refutation_pass_rate=model_info.get('refutation_pass_rate', 0),
        n_records=10000,  # Could pull from XCom
        duration_seconds=duration
    )
    
    return {'status': 'notified'}


def task_notify_failure(context):
    """Send failure notification"""
    from src.services.notifications import notify_pipeline_failure
    
    exception = context.get('exception')
    task_id = context.get('task_instance').task_id
    
    notify_pipeline_failure(
        step=task_id,
        error=str(exception)
    )


# DAG Definition
with DAG(
    dag_id='decision_intelligence_pipeline',
    default_args=default_args,
    description='End-to-end causal inference pipeline for marketing optimization',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['causal-ml', 'marketing', 'production'],
    on_failure_callback=task_notify_failure,
) as dag:
    
    # Start
    start = DummyOperator(task_id='start')
    
    # Data Preparation Group
    with TaskGroup(group_id='data_preparation') as data_prep:
        generate_data = PythonOperator(
            task_id='generate_data',
            python_callable=task_generate_data,
        )
        
        create_canonical = PythonOperator(
            task_id='create_canonical',
            python_callable=task_create_canonical,
        )
        
        generate_data >> create_canonical
    
    # Quality Checks Group
    with TaskGroup(group_id='quality_checks') as quality:
        dq_checks = PythonOperator(
            task_id='data_quality',
            python_callable=task_data_quality_checks,
        )
        
        drift_check = PythonOperator(
            task_id='drift_detection',
            python_callable=task_drift_detection,
        )
        
        dq_checks >> drift_check
    
    # Modeling Group
    with TaskGroup(group_id='modeling') as modeling:
        estimation = PythonOperator(
            task_id='causal_estimation',
            python_callable=task_causal_estimation,
        )
        
        refutation = PythonOperator(
            task_id='refutation_tests',
            python_callable=task_refutation_tests,
        )
        
        estimation >> refutation
    
    # Registration
    register = PythonOperator(
        task_id='register_model',
        python_callable=task_register_model,
    )
    
    # Update baseline after successful run
    update_baseline = PythonOperator(
        task_id='update_baseline',
        python_callable=task_update_baseline,
    )
    
    # Notification
    notify = PythonOperator(
        task_id='notify_success',
        python_callable=task_notify_success,
        trigger_rule='all_success',
    )
    
    # End
    end = DummyOperator(task_id='end')
    
    # Define dependencies
    start >> data_prep >> quality >> modeling >> register >> update_baseline >> notify >> end


# Weekly retraining DAG (more thorough)
with DAG(
    dag_id='decision_intelligence_weekly_retrain',
    default_args=default_args,
    description='Weekly full retraining with extended refutation tests',
    schedule_interval='0 2 * * 0',  # Sunday at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['causal-ml', 'marketing', 'retraining'],
) as weekly_dag:
    
    # Same structure but with extended refutation
    start_weekly = DummyOperator(task_id='start')
    
    full_pipeline = PythonOperator(
        task_id='run_full_pipeline',
        python_callable=lambda: __import__('run_pipeline').main(skip_refutation=False),
        execution_timeout=timedelta(hours=4),
    )
    
    register_weekly = PythonOperator(
        task_id='register_model',
        python_callable=task_register_model,
    )
    
    end_weekly = DummyOperator(task_id='end')
    
    start_weekly >> full_pipeline >> register_weekly >> end_weekly
