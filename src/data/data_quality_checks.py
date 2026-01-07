"""
Data Quality Checks for Causal Inference

Critical checks before causal estimation:
1. Required columns presence
2. Null rate thresholds
3. Unique key validation
4. Propensity score overlap (positivity assumption)
5. Covariate balance between treatment groups
6. Cardinality and distribution checks
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from scipy import stats
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    DATA_QUALITY,
    CANONICAL_SCHEMA,
    FILE_PATHS,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataQualityChecker:
    """Comprehensive data quality validation for causal inference"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {
            'overall_pass': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
    
    def run_all_checks(self) -> Dict:
        """Run all data quality checks"""
        logger.info("Running data quality checks...")
        
        # Critical checks
        self.check_required_columns()
        self.check_null_rates()
        self.check_unique_keys()
        self.check_treatment_support()
        self.check_propensity_overlap()
        self.check_covariate_balance()
        
        # Additional checks
        self.check_outcome_distribution()
        self.check_cardinality()
        
        # Summarize
        self._summarize_results()
        
        return self.results
    
    def check_required_columns(self) -> bool:
        """Check all required columns are present"""
        logger.info("  Checking required columns...")
        
        required_cols = CANONICAL_SCHEMA["required_columns"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            self.results['overall_pass'] = False
            self.results['errors'].append(f"Missing required columns: {missing_cols}")
            self.results['checks']['required_columns'] = {
                'pass': False,
                'missing': missing_cols
            }
            logger.error(f"  ✗ Missing columns: {missing_cols}")
            return False
        
        self.results['checks']['required_columns'] = {'pass': True}
        logger.info(f"  ✓ All {len(required_cols)} required columns present")
        return True
    
    def check_null_rates(self) -> bool:
        """Check null rates in critical columns"""
        logger.info("  Checking null rates...")
        
        max_null_rate = DATA_QUALITY["max_null_rate"]
        critical_cols = CANONICAL_SCHEMA["required_columns"]
        
        null_issues = {}
        for col in critical_cols:
            if col in self.df.columns:
                null_rate = self.df[col].isna().sum() / len(self.df)
                if null_rate > max_null_rate:
                    null_issues[col] = null_rate
        
        if null_issues:
            self.results['overall_pass'] = False
            self.results['errors'].append(
                f"Null rate exceeds {max_null_rate:.1%} threshold: {null_issues}"
            )
            self.results['checks']['null_rates'] = {
                'pass': False,
                'issues': null_issues
            }
            logger.error(f"  ✗ Null rate issues: {null_issues}")
            return False
        
        self.results['checks']['null_rates'] = {'pass': True}
        logger.info(f"  ✓ Null rates within acceptable limits")
        return True
    
    def check_unique_keys(self) -> bool:
        """Check uniqueness of user_id + event_ts"""
        logger.info("  Checking unique keys...")
        
        if 'user_id' not in self.df.columns or 'event_ts' not in self.df.columns:
            self.results['warnings'].append("Cannot check uniqueness: missing key columns")
            return True
        
        duplicates = self.df.duplicated(subset=['user_id', 'event_ts'], keep=False).sum()
        duplicate_rate = duplicates / len(self.df)
        
        if duplicates > 0:
            self.results['warnings'].append(
                f"Found {duplicates} ({duplicate_rate:.2%}) duplicate user_id+event_ts pairs"
            )
            self.results['checks']['unique_keys'] = {
                'pass': True,  # Warning, not failure
                'duplicates': int(duplicates),
                'rate': float(duplicate_rate)
            }
            logger.warning(f"  ⚠ Found {duplicates} duplicate keys")
        else:
            self.results['checks']['unique_keys'] = {'pass': True}
            logger.info(f"  ✓ All keys are unique")
        
        return True
    
    def check_treatment_support(self) -> bool:
        """Check sufficient support in both treatment groups (positivity)"""
        logger.info("  Checking treatment support...")
        
        if 'treatment' not in self.df.columns:
            return True
        
        treatment_counts = self.df['treatment'].value_counts()
        treatment_rates = treatment_counts / len(self.df)
        
        min_support = DATA_QUALITY["min_treatment_support"]
        
        insufficient_groups = []
        for treatment_val, rate in treatment_rates.items():
            if rate < min_support:
                insufficient_groups.append((treatment_val, rate))
        
        if insufficient_groups:
            self.results['overall_pass'] = False
            self.results['errors'].append(
                f"Insufficient treatment support: {insufficient_groups}"
            )
            self.results['checks']['treatment_support'] = {
                'pass': False,
                'insufficient': insufficient_groups
            }
            logger.error(f"  ✗ Insufficient support: {insufficient_groups}")
            return False
        
        self.results['checks']['treatment_support'] = {
            'pass': True,
            'rates': {int(k): float(v) for k, v in treatment_rates.items()}
        }
        logger.info(f"  ✓ Sufficient support in all groups")
        logger.info(f"    Treatment rates: {dict(treatment_rates)}")
        return True
    
    def check_propensity_overlap(self) -> bool:
        """
        Check propensity score overlap between treatment groups
        
        This is critical for causal inference - we need common support
        """
        logger.info("  Checking propensity score overlap...")
        
        if 'treatment' not in self.df.columns:
            return True
        
        # Get feature columns
        feature_cols = [col for col in CANONICAL_SCHEMA["feature_columns"] 
                       if col in self.df.columns]
        
        if len(feature_cols) == 0:
            self.results['warnings'].append("No features available for propensity check")
            return True
        
        # Simple propensity estimation using logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['treatment']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit propensity model
        prop_model = LogisticRegression(max_iter=1000, random_state=42)
        prop_model.fit(X_scaled, y)
        
        # Get propensity scores
        propensity_scores = prop_model.predict_proba(X_scaled)[:, 1]
        
        # Check overlap
        min_prop = DATA_QUALITY["min_propensity_overlap"]
        max_prop = DATA_QUALITY["max_propensity_overlap"]
        
        outside_overlap = (
            (propensity_scores < min_prop) | 
            (propensity_scores > max_prop)
        ).sum()
        outside_rate = outside_overlap / len(self.df)
        
        self.results['checks']['propensity_overlap'] = {
            'pass': True,
            'outside_overlap_count': int(outside_overlap),
            'outside_overlap_rate': float(outside_rate),
            'propensity_range': (float(propensity_scores.min()), 
                                float(propensity_scores.max())),
            'propensity_mean': float(propensity_scores.mean())
        }
        
        if outside_rate > 0.1:  # More than 10% outside overlap
            self.results['warnings'].append(
                f"{outside_rate:.1%} of samples have propensity scores outside [{min_prop}, {max_prop}]"
            )
            logger.warning(f"  ⚠ {outside_rate:.1%} outside overlap region")
        else:
            logger.info(f"  ✓ Good propensity overlap ({outside_rate:.1%} outside)")
        
        return True
    
    def check_covariate_balance(self) -> bool:
        """Check covariate balance between treatment groups"""
        logger.info("  Checking covariate balance...")
        
        if 'treatment' not in self.df.columns:
            return True
        
        feature_cols = [col for col in CANONICAL_SCHEMA["feature_columns"] 
                       if col in self.df.columns]
        
        if len(feature_cols) == 0:
            return True
        
        balance_results = {}
        imbalanced_features = []
        
        for col in feature_cols:
            # Skip if too many nulls
            if self.df[col].isna().sum() / len(self.df) > 0.2:
                continue
            
            treated = self.df[self.df['treatment'] == 1][col].dropna()
            control = self.df[self.df['treatment'] == 0][col].dropna()
            
            if len(treated) == 0 or len(control) == 0:
                continue
            
            # Standardized mean difference
            smd = abs(treated.mean() - control.mean()) / np.sqrt(
                (treated.var() + control.var()) / 2
            )
            
            # KS test for distribution difference
            ks_stat, ks_pval = stats.ks_2samp(treated, control)
            
            balance_results[col] = {
                'smd': float(smd),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval)
            }
            
            # Flag if significantly imbalanced
            if smd > 0.1 or ks_pval < 0.01:
                imbalanced_features.append((col, smd, ks_stat))
        
        self.results['checks']['covariate_balance'] = {
            'pass': True,
            'balance_results': balance_results,
            'imbalanced_features': len(imbalanced_features)
        }
        
        if imbalanced_features:
            self.results['warnings'].append(
                f"{len(imbalanced_features)} features show imbalance between groups"
            )
            logger.warning(f"  ⚠ {len(imbalanced_features)} imbalanced features")
            for feat, smd, ks in imbalanced_features[:3]:  # Show top 3
                logger.warning(f"    {feat}: SMD={smd:.3f}, KS={ks:.3f}")
        else:
            logger.info(f"  ✓ Good covariate balance across groups")
        
        return True
    
    def check_outcome_distribution(self) -> bool:
        """Check outcome variable distribution"""
        logger.info("  Checking outcome distribution...")
        
        if 'outcome' not in self.df.columns:
            return True
        
        outcome = self.df['outcome'].dropna()
        
        stats_dict = {
            'count': int(len(outcome)),
            'mean': float(outcome.mean()),
            'std': float(outcome.std()),
            'min': float(outcome.min()),
            'max': float(outcome.max()),
            'negative_count': int((outcome < 0).sum()),
            'zero_count': int((outcome == 0).sum()),
        }
        
        self.results['checks']['outcome_distribution'] = {
            'pass': True,
            'stats': stats_dict
        }
        
        logger.info(f"  ✓ Outcome stats: mean=${stats_dict['mean']:.2f}, "
                   f"std=${stats_dict['std']:.2f}")
        
        if stats_dict['negative_count'] > 0:
            self.results['warnings'].append(
                f"{stats_dict['negative_count']} negative outcome values"
            )
        
        return True
    
    def check_cardinality(self) -> bool:
        """Check cardinality of categorical features"""
        logger.info("  Checking feature cardinality...")
        
        feature_cols = [col for col in CANONICAL_SCHEMA["feature_columns"] 
                       if col in self.df.columns]
        
        cardinality_results = {}
        high_cardinality = []
        
        for col in feature_cols:
            unique_count = self.df[col].nunique()
            cardinality_results[col] = int(unique_count)
            
            # Flag very high cardinality (might be ID column by mistake)
            if unique_count > len(self.df) * 0.9:
                high_cardinality.append((col, unique_count))
        
        self.results['checks']['cardinality'] = {
            'pass': True,
            'cardinality': cardinality_results
        }
        
        if high_cardinality:
            self.results['warnings'].append(
                f"Very high cardinality features (possible ID columns): {high_cardinality}"
            )
            logger.warning(f"  ⚠ High cardinality: {[c[0] for c in high_cardinality]}")
        else:
            logger.info(f"  ✓ Cardinality check passed")
        
        return True
    
    def _summarize_results(self):
        """Summarize all check results"""
        total_checks = len(self.results['checks'])
        passed_checks = sum(1 for c in self.results['checks'].values() if c['pass'])
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY CHECK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}/{total_checks}")
        logger.info(f"Warnings: {len(self.results['warnings'])}")
        logger.info(f"Errors: {len(self.results['errors'])}")
        
        if self.results['overall_pass']:
            logger.info("✓ OVERALL: PASS")
        else:
            logger.error("✗ OVERALL: FAIL")
            for error in self.results['errors']:
                logger.error(f"  - {error}")
        
        if self.results['warnings']:
            logger.warning("\nWarnings:")
            for warning in self.results['warnings']:
                logger.warning(f"  - {warning}")


def load_data(file_path: Path) -> pd.DataFrame:
    """Load canonical data"""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    logger.info(f"✓ Loaded {len(df)} records")
    return df


def save_report(results: Dict, output_path: Path):
    """Save DQ report to JSON"""
    logger.info(f"Saving DQ report to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Report saved")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("STEP 3: DATA QUALITY CHECKS")
    logger.info("=" * 60)
    
    # Load data
    df = load_data(FILE_PATHS["canonical_data"])
    
    # Run checks
    checker = DataQualityChecker(df)
    results = checker.run_all_checks()
    
    # Save report
    save_report(results, FILE_PATHS["data_quality_report"])
    
    logger.info("=" * 60)
    
    if not results['overall_pass']:
        logger.error("✗ Data quality checks failed. Fix issues before proceeding.")
        return False
    
    logger.info("✓ Data quality checks passed!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)