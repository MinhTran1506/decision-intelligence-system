"""
Transform raw data into canonical schema for causal analysis

This module implements the ETL process that:
1. Loads raw marketing event data
2. Validates and cleanses data
3. Creates canonical schema expected by downstream pipeline
4. Saves processed data for causal estimation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import CANONICAL_SCHEMA, FILE_PATHS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw marketing events data"""
    logger.info(f"Loading raw data from {file_path}...")
    df = pd.read_parquet(file_path)
    logger.info(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def create_canonical_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data to canonical schema
    
    Canonical schema ensures:
    - Consistent column names across pipeline
    - Required columns present
    - Feature columns properly encoded
    - Data types correct
    
    Args:
        df: Raw data DataFrame
        
    Returns:
        Canonicalized DataFrame
    """
    logger.info("Creating canonical dataset...")
    
    # Select and rename columns to match canonical schema
    canonical_df = df.copy()
    
    # Ensure required columns exist
    required_cols = CANONICAL_SCHEMA["required_columns"]
    for col in required_cols:
        if col not in canonical_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Validate data types
    canonical_df['user_id'] = canonical_df['user_id'].astype(str)
    canonical_df['treatment'] = canonical_df['treatment'].astype(int)
    canonical_df['outcome'] = canonical_df['outcome'].astype(float)
    canonical_df['event_ts'] = pd.to_datetime(canonical_df['event_ts'])
    
    # Ensure feature columns are numeric
    feature_cols = CANONICAL_SCHEMA["feature_columns"]
    for col in feature_cols:
        if col in canonical_df.columns:
            canonical_df[col] = pd.to_numeric(canonical_df[col], errors='coerce')
    
    # Create cohort_id (optional) - weekly cohorts
    canonical_df['cohort_id'] = canonical_df['event_ts'].dt.to_period('W').astype(str)
    
    # Add metadata
    canonical_df['canonical_created_at'] = pd.Timestamp.now()
    
    # Select final columns in order
    final_columns = (
        required_cols +
        CANONICAL_SCHEMA.get("optional_columns", []) +
        [col for col in feature_cols if col in canonical_df.columns] +
        ['source_table', 'ingested_at', 'canonical_created_at']
    )
    
    # Keep only columns that exist
    final_columns = [col for col in final_columns if col in canonical_df.columns]
    canonical_df = canonical_df[final_columns]
    
    logger.info(f"✓ Created canonical dataset with {len(canonical_df)} records")
    logger.info(f"  Required columns: {required_cols}")
    logger.info(f"  Feature columns: {len(feature_cols)} features")
    
    return canonical_df


def validate_canonical_dataset(df: pd.DataFrame) -> dict:
    """
    Validate canonical dataset meets requirements
    
    Returns:
        dict: Validation results
    """
    logger.info("Validating canonical dataset...")
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    required_cols = CANONICAL_SCHEMA["required_columns"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check for nulls in required columns
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_rate = null_count / len(df)
            if null_rate > 0:
                validation_results['warnings'].append(
                    f"Column '{col}' has {null_rate:.1%} null values"
                )
    
    # Check treatment values
    if 'treatment' in df.columns:
        unique_treatments = df['treatment'].unique()
        if not all(t in [0, 1] for t in unique_treatments):
            validation_results['errors'].append(
                f"Treatment column contains invalid values: {unique_treatments}"
            )
    
    # Check outcome values
    if 'outcome' in df.columns:
        if (df['outcome'] < 0).any():
            validation_results['warnings'].append(
                "Outcome column contains negative values"
            )
    
    # Collect statistics
    validation_results['stats'] = {
        'n_records': len(df),
        'n_features': len([col for col in df.columns 
                          if col in CANONICAL_SCHEMA['feature_columns']]),
        'treatment_rate': df['treatment'].mean() if 'treatment' in df.columns else None,
        'mean_outcome': df['outcome'].mean() if 'outcome' in df.columns else None,
        'date_range': (
            df['event_ts'].min(), df['event_ts'].max()
        ) if 'event_ts' in df.columns else None,
    }
    
    # Log results
    if validation_results['valid']:
        logger.info("✓ Validation passed")
    else:
        logger.error(f"✗ Validation failed: {validation_results['errors']}")
    
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            logger.warning(f"  ⚠ {warning}")
    
    # Log stats
    logger.info("Dataset statistics:")
    for key, value in validation_results['stats'].items():
        logger.info(f"  {key}: {value}")
    
    return validation_results


def save_canonical_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save canonical dataset"""
    logger.info(f"Saving canonical dataset to {output_path}...")
    df.to_parquet(output_path, index=False, compression='snappy')
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Canonical dataset saved ({size_mb:.2f} MB)")


def print_sample_records(df: pd.DataFrame, n: int = 5) -> None:
    """Print sample records for inspection"""
    logger.info(f"\nSample canonical records (first {n}):")
    
    display_cols = [
        'user_id', 'event_ts', 'treatment', 'outcome',
        'age', 'income_level', 'engagement_score', 'cohort_id'
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    
    print("\n" + df[display_cols].head(n).to_string(index=False))


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("STEP 2: CREATE CANONICAL DATASET")
    logger.info("=" * 60)
    
    # Load raw data
    raw_df = load_raw_data(FILE_PATHS["raw_data"])
    
    # Create canonical dataset
    canonical_df = create_canonical_dataset(raw_df)
    
    # Validate
    validation_results = validate_canonical_dataset(canonical_df)
    
    if not validation_results['valid']:
        logger.error("Canonical dataset validation failed. Aborting.")
        return False
    
    # Save
    save_canonical_dataset(canonical_df, FILE_PATHS["canonical_data"])
    
    # Print sample
    print_sample_records(canonical_df)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Canonical dataset creation complete!")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)