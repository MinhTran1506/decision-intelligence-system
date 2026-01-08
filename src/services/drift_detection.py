"""
Drift Detection Service

Monitor feature distributions and model performance over time.
Detect data drift using statistical tests (KS test, PSI).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    DATA_QUALITY, 
    CANONICAL_SCHEMA, 
    FILE_PATHS, 
    PROJECT_ROOT
)
from src.utils.logging_config import get_logger
from src.services.notifications import get_notifier

logger = get_logger(__name__)

DRIFT_REPORT_PATH = PROJECT_ROOT / "data" / "outputs" / "drift_report.json"
BASELINE_STATS_PATH = PROJECT_ROOT / "data" / "outputs" / "baseline_stats.json"


class DriftDetector:
    """
    Detect distribution drift in features and model outputs
    
    Uses:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-square test for categorical features
    - Population Stability Index (PSI) for score distributions
    """
    
    def __init__(self, baseline_df: pd.DataFrame = None):
        self.baseline_df = baseline_df
        self.baseline_stats = {}
        self.drift_results = {}
        self.ks_threshold = DATA_QUALITY["max_ks_statistic"]
        self.notifier = get_notifier()
    
    def compute_baseline_stats(self, df: pd.DataFrame) -> Dict:
        """Compute and save baseline statistics for drift detection"""
        logger.info("Computing baseline statistics...")
        
        stats_dict = {
            'computed_at': datetime.now().isoformat(),
            'n_records': len(df),
            'features': {}
        }
        
        feature_cols = [col for col in CANONICAL_SCHEMA["feature_columns"] 
                       if col in df.columns]
        
        for col in feature_cols:
            col_data = df[col].dropna()
            
            if col_data.dtype in ['float64', 'int64', 'float32', 'int32']:
                stats_dict['features'][col] = {
                    'type': 'numeric',
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'median': float(col_data.median()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'percentiles': {
                        '25': float(col_data.quantile(0.25)),
                        '50': float(col_data.quantile(0.50)),
                        '75': float(col_data.quantile(0.75)),
                        '95': float(col_data.quantile(0.95))
                    }
                }
            else:
                # Categorical
                value_counts = col_data.value_counts(normalize=True).to_dict()
                stats_dict['features'][col] = {
                    'type': 'categorical',
                    'distribution': {str(k): float(v) for k, v in value_counts.items()},
                    'n_unique': int(col_data.nunique())
                }
        
        # Add outcome and treatment stats
        if 'outcome' in df.columns:
            stats_dict['outcome'] = {
                'mean': float(df['outcome'].mean()),
                'std': float(df['outcome'].std())
            }
        
        if 'treatment' in df.columns:
            stats_dict['treatment_rate'] = float(df['treatment'].mean())
        
        # Save baseline
        with open(BASELINE_STATS_PATH, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        self.baseline_stats = stats_dict
        logger.info(f"Baseline stats saved with {len(feature_cols)} features")
        
        return stats_dict
    
    def load_baseline_stats(self) -> Dict:
        """Load previously computed baseline statistics"""
        if BASELINE_STATS_PATH.exists():
            with open(BASELINE_STATS_PATH, 'r') as f:
                self.baseline_stats = json.load(f)
            logger.info("Loaded baseline statistics from file")
            return self.baseline_stats
        else:
            logger.warning("No baseline statistics found")
            return {}
    
    def ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test
        
        Returns:
            (ks_statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        return float(statistic), float(p_value)
    
    def psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change
        """
        # Bin the distributions
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        
        baseline_counts, _ = np.histogram(baseline, bins=bins_edges)
        current_counts, _ = np.histogram(current, bins=bins_edges)
        
        # Convert to proportions
        baseline_prop = (baseline_counts + 1) / (len(baseline) + bins)  # Add smoothing
        current_prop = (current_counts + 1) / (len(current) + bins)
        
        # Calculate PSI
        psi_value = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
        
        return float(psi_value)
    
    def detect_drift(
        self, 
        current_df: pd.DataFrame,
        alert_on_drift: bool = True
    ) -> Dict:
        """
        Detect drift between baseline and current data
        
        Args:
            current_df: Current data to compare against baseline
            alert_on_drift: Whether to send alerts for detected drift
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Running drift detection...")
        
        if not self.baseline_stats:
            self.load_baseline_stats()
        
        if not self.baseline_stats:
            logger.warning("No baseline stats available. Computing from current data.")
            self.compute_baseline_stats(current_df)
            return {'message': 'Baseline computed, no drift detected (first run)'}
        
        results = {
            'run_timestamp': datetime.now().isoformat(),
            'n_records_current': len(current_df),
            'n_records_baseline': self.baseline_stats.get('n_records', 0),
            'drift_detected': False,
            'features_with_drift': [],
            'feature_results': {}
        }
        
        # Check each feature
        for col, baseline_info in self.baseline_stats.get('features', {}).items():
            if col not in current_df.columns:
                continue
            
            current_data = current_df[col].dropna()
            
            if baseline_info['type'] == 'numeric':
                # KS test for numeric features
                baseline_mean = baseline_info['mean']
                baseline_std = baseline_info['std']
                
                # Generate baseline samples from stored stats (approximation)
                baseline_samples = np.random.normal(
                    baseline_mean, 
                    baseline_std, 
                    size=min(10000, len(current_data))
                )
                
                ks_stat, p_value = self.ks_test(baseline_samples, current_data.values)
                
                drift_detected = ks_stat > self.ks_threshold
                
                results['feature_results'][col] = {
                    'type': 'numeric',
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'threshold': self.ks_threshold,
                    'drift_detected': drift_detected,
                    'baseline_mean': baseline_mean,
                    'current_mean': float(current_data.mean()),
                    'mean_shift': float(current_data.mean() - baseline_mean)
                }
                
                if drift_detected:
                    results['drift_detected'] = True
                    results['features_with_drift'].append(col)
                    
                    logger.warning(f"Drift detected in {col}: KS={ks_stat:.4f}")
                    
                    if alert_on_drift:
                        self.notifier.send_drift_alert(
                            feature=col,
                            ks_statistic=ks_stat,
                            threshold=self.ks_threshold,
                            baseline_mean=baseline_mean,
                            current_mean=float(current_data.mean())
                        )
            
            else:
                # Chi-square test for categorical features
                baseline_dist = baseline_info.get('distribution', {})
                current_dist = current_data.value_counts(normalize=True).to_dict()
                
                # Calculate distribution difference
                all_categories = set(baseline_dist.keys()) | set(current_dist.keys())
                total_diff = 0
                
                for cat in all_categories:
                    b = baseline_dist.get(str(cat), 0)
                    c = current_dist.get(cat, 0)
                    total_diff += abs(b - c)
                
                drift_detected = total_diff > 0.2  # 20% distribution shift
                
                results['feature_results'][col] = {
                    'type': 'categorical',
                    'distribution_shift': total_diff,
                    'drift_detected': drift_detected
                }
                
                if drift_detected:
                    results['drift_detected'] = True
                    results['features_with_drift'].append(col)
        
        # Check uplift score distribution if available
        if 'uplift_score' in current_df.columns and 'outcome' in self.baseline_stats:
            current_mean = current_df['uplift_score'].mean()
            baseline_mean = self.baseline_stats['outcome']['mean']
            
            psi_value = self.psi(
                np.random.normal(baseline_mean, self.baseline_stats['outcome']['std'], 10000),
                current_df['uplift_score'].values
            )
            
            results['uplift_score_psi'] = psi_value
            results['uplift_drift'] = psi_value > 0.25
        
        # Save results
        self.drift_results = results
        with open(DRIFT_REPORT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary log
        if results['drift_detected']:
            logger.warning(f"⚠️ Drift detected in {len(results['features_with_drift'])} features")
        else:
            logger.info("✓ No significant drift detected")
        
        return results
    
    def get_drift_summary(self) -> str:
        """Get human-readable drift summary"""
        if not self.drift_results:
            return "No drift detection results available"
        
        lines = [
            "DRIFT DETECTION SUMMARY",
            "=" * 40,
            f"Timestamp: {self.drift_results.get('run_timestamp', 'N/A')}",
            f"Records: {self.drift_results.get('n_records_current', 0):,}",
            f"Drift Detected: {'Yes' if self.drift_results.get('drift_detected') else 'No'}",
        ]
        
        if self.drift_results.get('features_with_drift'):
            lines.append(f"\nFeatures with drift:")
            for feat in self.drift_results['features_with_drift']:
                feat_info = self.drift_results['feature_results'].get(feat, {})
                ks = feat_info.get('ks_statistic', 0)
                lines.append(f"  - {feat}: KS={ks:.4f}")
        
        return "\n".join(lines)


def run_drift_detection(current_data_path: Path = None) -> Dict:
    """
    Run drift detection as standalone function
    
    Args:
        current_data_path: Path to current data. If None, uses canonical data.
    """
    if current_data_path is None:
        current_data_path = FILE_PATHS["canonical_data"]
    
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION")
    logger.info("=" * 60)
    
    # Load current data
    current_df = pd.read_parquet(current_data_path)
    
    # Run detection
    detector = DriftDetector()
    results = detector.detect_drift(current_df)
    
    print(detector.get_drift_summary())
    
    return results


def update_baseline(data_path: Path = None):
    """Update baseline statistics from current data"""
    if data_path is None:
        data_path = FILE_PATHS["canonical_data"]
    
    df = pd.read_parquet(data_path)
    
    detector = DriftDetector()
    detector.compute_baseline_stats(df)
    
    logger.info("Baseline statistics updated")


if __name__ == "__main__":
    run_drift_detection()
