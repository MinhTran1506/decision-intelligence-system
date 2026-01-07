"""
Configuration file for Decision Intelligence Studio
"""
import os
from pathlib import Path
from typing import List, Dict, Any

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data Generation Parameters
DATA_CONFIG = {
    "sample_size": 10000,
    "treatment_propensity_base": 0.3,  # 30% baseline treatment probability
    "random_seed": 42,
    "date_range_days": 90,
    "true_ate": 45.0,  # True average treatment effect for validation
    "noise_level": 0.2,
}

# Feature Configuration
FEATURES = {
    "demographic": ["age", "income_level", "region"],
    "behavioral": ["past_purchases", "days_since_signup", "engagement_score"],
    "contextual": ["season", "day_of_week"],
}

# Canonical Schema Definition
CANONICAL_SCHEMA = {
    "required_columns": [
        "user_id",
        "event_ts",
        "treatment",
        "outcome",
    ],
    "optional_columns": [
        "treatment_val",
        "outcome_window",
        "cohort_id",
    ],
    "feature_columns": [
        "age",
        "income_level",
        "region_encoded",
        "past_purchases",
        "days_since_signup",
        "engagement_score",
        "season_encoded",
        "day_of_week",
    ],
}

# Data Quality Thresholds
DATA_QUALITY = {
    "max_null_rate": 0.05,  # 5% maximum nulls
    "min_treatment_support": 0.1,  # At least 10% in each group
    "min_propensity_overlap": 0.05,  # Minimum propensity score
    "max_propensity_overlap": 0.95,  # Maximum propensity score
    "max_ks_statistic": 0.15,  # Max KS statistic for drift
}

# Causal Graph Definition
CAUSAL_GRAPH = {
    "nodes": [
        "age",
        "income_level",
        "past_purchases",
        "days_since_signup",
        "engagement_score",
        "treatment",
        "outcome",
    ],
    "edges": [
        # Confounders -> Treatment
        ["age", "treatment"],
        ["income_level", "treatment"],
        ["past_purchases", "treatment"],
        ["engagement_score", "treatment"],
        # Confounders -> Outcome
        ["age", "outcome"],
        ["income_level", "outcome"],
        ["past_purchases", "outcome"],
        ["days_since_signup", "outcome"],
        ["engagement_score", "outcome"],
        # Treatment -> Outcome (causal effect of interest)
        ["treatment", "outcome"],
    ],
    "adjustment_set": [
        "age",
        "income_level",
        "past_purchases",
        "engagement_score",
    ],
}

# Model Configuration
MODEL_CONFIG = {
    "causal_forest": {
        "n_estimators": 1000,
        "min_samples_leaf": 50,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    },
    "base_models": {
        "model_y": {  # Outcome model
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 50,
            "random_state": 42,
        },
        "model_t": {  # Treatment model
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_split": 50,
            "random_state": 42,
        },
    },
    "cross_fitting_folds": 2,
}

# Refutation Test Configuration
REFUTATION_CONFIG = {
    "placebo_test": {
        "enabled": True,
        "num_simulations": 100,
    },
    "random_common_cause": {
        "enabled": True,
    },
    "subset_validation": {
        "enabled": True,
        "test_fraction": 0.3,
    },
    "data_subset": {
        "enabled": True,
        "subset_fraction": 0.8,
    },
    "bootstrap": {
        "enabled": True,
        "num_simulations": 50,
    },
}

# Business Rules & Decision Logic
DECISION_RULES = {
    "min_uplift_threshold": 5.0,  # Minimum uplift in dollars
    "expected_value_per_conversion": 1.0,  # Multiplier for uplift
    "cost_per_treatment": 10.0,  # Cost to send promotion
    "min_roi_threshold": 0.5,  # 50% minimum ROI
    "budget_limit": 50000.0,  # Total campaign budget
    "max_users_per_campaign": 5000,
}

# Segmentation Configuration
SEGMENTATION = {
    "method": "quantile",  # "quantile" or "threshold"
    "num_segments": 4,
    "segment_names": ["High Uplift", "Medium-High", "Medium-Low", "Low Uplift"],
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "Decision Intelligence Studio API",
    "description": "Causal inference and uplift modeling API",
    "version": "1.0.0",
    "enable_cors": True,
}

# Monitoring & Alerting
MONITORING = {
    "enable_metrics": True,
    "log_level": "INFO",
    "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
    "alert_on_failure": True,
    "performance_metrics": [
        "ate_estimate",
        "refutation_pass_rate",
        "uplift_score_distribution",
        "data_quality_score",
    ],
}

# Model Registry
MODEL_REGISTRY = {
    "model_version": "v1.0",
    "model_name": "causal_forest_marketing",
    "training_date": None,  # Will be set at runtime
    "git_commit": os.getenv("GIT_COMMIT", "local"),
    "docker_image": os.getenv("DOCKER_IMAGE", "local"),
}

# File Paths (computed)
FILE_PATHS = {
    "raw_data": RAW_DATA_DIR / "marketing_events.parquet",
    "canonical_data": PROCESSED_DATA_DIR / "canonical_events.parquet",
    "uplift_scores": OUTPUTS_DIR / "uplift_scores.parquet",
    "model_artifact": MODELS_DIR / "causal_forest_v1.joblib",
    "refutation_report": OUTPUTS_DIR / "refutation_report.json",
    "data_quality_report": OUTPUTS_DIR / "data_quality_report.json",
    "causal_graph": PROJECT_ROOT / "src" / "causal" / "causal_graph.json",
}


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "data": DATA_CONFIG,
        "features": FEATURES,
        "schema": CANONICAL_SCHEMA,
        "data_quality": DATA_QUALITY,
        "causal_graph": CAUSAL_GRAPH,
        "model": MODEL_CONFIG,
        "refutation": REFUTATION_CONFIG,
        "decision_rules": DECISION_RULES,
        "segmentation": SEGMENTATION,
        "api": API_CONFIG,
        "monitoring": MONITORING,
        "model_registry": MODEL_REGISTRY,
        "file_paths": FILE_PATHS,
    }


def print_config():
    """Print current configuration (for debugging)"""
    import json
    
    config = get_config()
    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    print_config()