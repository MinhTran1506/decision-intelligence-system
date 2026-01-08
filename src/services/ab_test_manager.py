"""
A/B Test Manager Service

Provides complete A/B test lifecycle management:
- Create and configure experiments
- Random assignment of users to groups  
- Track conversions and outcomes
- Calculate statistical significance
- Generate reports
"""
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.data_store import get_store
from src.utils.config import FILE_PATHS, SEGMENTATION
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ABTestManager:
    """Manages A/B test experiments"""
    
    def __init__(self):
        self.store = get_store()
        self._load_uplift_data()
    
    def _load_uplift_data(self):
        """Load uplift data for predictions"""
        try:
            self.uplift_data = pd.read_parquet(FILE_PATHS["uplift_scores"])
        except Exception as e:
            logger.warning(f"Could not load uplift data: {e}")
            self.uplift_data = None
    
    def create_experiment(
        self,
        name: str,
        segment: str,
        sample_size: int,
        description: str = "",
        control_ratio: float = 0.5,
        hypothesis: str = "",
        test_id: Optional[str] = None,
        predicted_uplift: Optional[float] = None
    ) -> bool:
        """Create a new A/B test experiment
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Generate unique test ID if not provided
        if test_id is None:
            test_id = f"TEST-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get predicted uplift from model if not provided
        if predicted_uplift is None:
            predicted_uplift = self._get_predicted_uplift(segment)
        
        # Calculate required sample size for significance
        min_sample = self._calculate_min_sample_size(predicted_uplift)
        
        if sample_size < min_sample:
            logger.warning(
                f"Sample size {sample_size} may be too small for significance. "
                f"Recommended: {min_sample}"
            )
        
        config = {
            "hypothesis": hypothesis,
            "min_sample_for_significance": min_sample,
            "expected_runtime_days": self._estimate_runtime(sample_size),
        }
        
        success = self.store.create_ab_test(
            test_id=test_id,
            name=name,
            segment=segment,
            sample_size=sample_size,
            description=description,
            control_ratio=control_ratio,
            predicted_uplift=predicted_uplift,
            config=config
        )
        
        if success:
            logger.info(f"Created A/B test: {test_id}")
            return True
        else:
            logger.error(f"Failed to create test {test_id}")
            return False
    
    def _get_predicted_uplift(self, segment: str) -> float:
        """Get predicted uplift for segment from model"""
        if self.uplift_data is None:
            return 45.0  # Default estimate
        
        if segment == "All Users":
            return float(self.uplift_data['uplift_score'].mean())
        
        seg_data = self.uplift_data[self.uplift_data['segment_name'] == segment]
        if len(seg_data) > 0:
            return float(seg_data['uplift_score'].mean())
        
        return 45.0
    
    def _calculate_min_sample_size(
        self, 
        expected_effect: float,
        alpha: float = 0.05,
        power: float = 0.8,
        baseline_rate: float = 0.1
    ) -> int:
        """Calculate minimum sample size for statistical power"""
        # Convert uplift to relative effect
        relative_effect = expected_effect / 100  # Assuming $100 baseline
        
        # Effect size (Cohen's h for proportions)
        p1 = baseline_rate
        p2 = baseline_rate * (1 + relative_effect)
        p2 = min(p2, 0.99)  # Cap at 99%
        
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        if abs(h) < 0.01:
            return 10000  # Very small effect needs large sample
        
        # Sample size calculation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / h) ** 2
        
        return max(int(np.ceil(n)), 100)
    
    def _estimate_runtime(self, sample_size: int, events_per_day: int = 500) -> int:
        """Estimate how many days to reach sample size"""
        return max(1, int(np.ceil(sample_size / events_per_day)))
    
    def start_experiment(self, test_id: str) -> bool:
        """Start an experiment
        
        Returns:
            bool: True if successful, False otherwise
        """
        test = self.store.get_ab_test(test_id)
        
        if not test:
            logger.error(f"Test {test_id} not found")
            return False
        
        if test['status'] != 'draft':
            logger.error(f"Test {test_id} is already {test['status']}")
            return False
        
        success = self.store.start_ab_test(test_id)
        
        if success:
            logger.info(f"Started experiment {test_id}")
            return True
        
        return False
    
    def assign_user(self, test_id: str, user_id: str) -> Optional[str]:
        """Assign a user to a test group using deterministic hashing"""
        test = self.store.get_ab_test(test_id)
        
        if not test or test['status'] != 'running':
            return None
        
        # Check if already assigned
        existing = self.store.get_user_test_assignment(test_id, user_id)
        if existing:
            return existing
        
        # Deterministic assignment using hash
        hash_input = f"{test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        ratio = (hash_value % 1000) / 1000.0
        
        group = "control" if ratio < test['control_ratio'] else "treatment"
        
        # Record assignment
        self.store.assign_user_to_test(test_id, user_id, group)
        
        return group
    
    def record_conversion(
        self, 
        test_id: str, 
        user_id: str, 
        value: float = 1.0,
        event_type: str = "conversion"
    ) -> bool:
        """Record a conversion event for a user in a test
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Check user is assigned
        group = self.store.get_user_test_assignment(test_id, user_id)
        
        if not group:
            return False
        
        # Record event
        self.store.record_test_event(
            test_id=test_id,
            user_id=user_id,
            event_type=event_type,
            event_value=value,
            metadata={"group": group}
        )
        
        return True
    
    def get_experiment_results(self, test_id: str) -> Dict:
        """Get current results for an experiment"""
        test = self.store.get_ab_test(test_id)
        
        if not test:
            return {"status": "error", "message": "Test not found"}
        
        results = self.store.get_test_results(test_id)
        
        # Add test metadata
        results['test_name'] = test['name']
        results['segment'] = test['segment']
        results['status'] = test['status']
        results['predicted_uplift'] = test['predicted_uplift']
        results['start_date'] = test['start_date']
        results['end_date'] = test['end_date']
        
        # Calculate additional metrics
        if results['lift'] is not None:
            results['prediction_error'] = abs(
                (results['lift'] / 100 * 100) - (test['predicted_uplift'] or 0)
            )
        
        return results
    
    def stop_experiment(self, test_id: str) -> bool:
        """Stop an experiment and finalize results
        
        Returns:
            bool: True if successful, False otherwise
        """
        results = self.get_experiment_results(test_id)
        
        if 'error' in results:
            logger.error(f"Cannot stop test {test_id}: {results.get('error')}")
            return False
        
        # Calculate final metrics
        observed_uplift = results.get('lift', 0) or 0
        
        # Calculate p-value properly
        p_value = self._calculate_p_value(results)
        
        success = self.store.stop_ab_test(
            test_id=test_id,
            observed_uplift=observed_uplift,
            p_value=p_value or 1.0
        )
        
        if success:
            logger.info(f"Stopped experiment {test_id}, observed uplift: {observed_uplift}%")
            return True
        
        return False
    
    def _calculate_p_value(self, results: Dict) -> Optional[float]:
        """Calculate p-value using chi-square test"""
        groups = results.get('groups', {})
        
        if 'control' not in groups or 'treatment' not in groups:
            return None
        
        control = groups['control']
        treatment = groups['treatment']
        
        # Build contingency table
        control_conv = control.get('conversions', 0)
        control_no_conv = control.get('users', 0) - control_conv
        treatment_conv = treatment.get('conversions', 0)
        treatment_no_conv = treatment.get('users', 0) - treatment_conv
        
        if control_no_conv < 0 or treatment_no_conv < 0:
            return None
        
        # Chi-square test
        table = np.array([
            [control_conv, control_no_conv],
            [treatment_conv, treatment_no_conv]
        ])
        
        if np.any(table < 5):
            # Use Fisher's exact test for small samples
            try:
                _, p_value = stats.fisher_exact(table)
                return float(p_value)
            except:
                return None
        
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            return float(p_value)
        except:
            return None
    
    def _get_recommendation(self, results: Dict, p_value: Optional[float]) -> str:
        """Generate recommendation based on results"""
        if p_value is None:
            return "Insufficient data to make a recommendation."
        
        lift = results.get('lift', 0) or 0
        
        if p_value >= 0.05:
            return (
                "Results are NOT statistically significant (p >= 0.05). "
                "Consider running the test longer or increasing sample size."
            )
        
        if lift > 0:
            return (
                f"Treatment shows a significant {lift:.1f}% improvement (p < 0.05). "
                "Recommend rolling out the treatment to all users."
            )
        else:
            return (
                f"Treatment shows a significant {abs(lift):.1f}% decrease (p < 0.05). "
                "Recommend keeping the control version."
            )
    
    def list_experiments(self, status: Optional[str] = None) -> List[Dict]:
        """List all experiments"""
        return self.store.list_ab_tests(status)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all tests"""
        all_tests = self.list_experiments()
        
        completed = [t for t in all_tests if t['status'] == 'completed']
        running = [t for t in all_tests if t['status'] == 'running']
        draft = [t for t in all_tests if t['status'] == 'draft']
        
        avg_error = None
        if completed:
            errors = []
            for t in completed:
                if t['predicted_uplift'] and t['observed_uplift']:
                    errors.append(abs(t['predicted_uplift'] - t['observed_uplift']))
            if errors:
                avg_error = np.mean(errors)
        
        return {
            "total_tests": len(all_tests),
            "completed": len(completed),
            "running": len(running),
            "draft": len(draft),
            "avg_prediction_error": avg_error,
            "model_calibration": f"{100 - avg_error:.1f}%" if avg_error else "N/A"
        }


# Singleton instance
_manager_instance = None

def get_ab_manager() -> ABTestManager:
    """Get or create ABTestManager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ABTestManager()
    return _manager_instance
