"""
Refutation Tests for Causal Estimates

Implements robustness checks to validate causal estimates:
1. Placebo treatment test - shuffle treatment, expect ~0 effect
2. Random common cause - add random variable, check stability
3. Subset validation - estimate on subset, compare
4. Data subset refutation - use portion of data
5. Bootstrap validation - check confidence intervals
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, List
from joblib import load

# Causal inference libraries
from dowhy import CausalModel

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    REFUTATION_CONFIG,
    CANONICAL_SCHEMA,
    FILE_PATHS,
    CAUSAL_GRAPH,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RefutationTester:
    """
    Comprehensive refutation testing for causal estimates
    
    These tests help validate our causal conclusions by checking
    if they hold up under various perturbations
    """
    
    def __init__(self, df: pd.DataFrame, causal_model: CausalModel = None):
        self.df = df
        self.causal_model = causal_model
        self.results = {
            'overall_pass': True,
            'tests': {},
            'summary': {}
        }
    
    def build_causal_model_if_needed(self) -> CausalModel:
        """Build causal model if not provided"""
        if self.causal_model is not None:
            return self.causal_model
        
        logger.info("Building causal model for refutation...")
        
        # Get feature columns
        feature_cols = [col for col in CANONICAL_SCHEMA["feature_columns"] 
                       if col in self.df.columns]
        
        # Prepare data
        data_for_dowhy = self.df[feature_cols + ['treatment', 'outcome']].copy()
        data_for_dowhy = data_for_dowhy.fillna(0)
        
        # Build graph
        edges = CAUSAL_GRAPH["edges"]
        dot_lines = ["digraph {"]
        for source, target in edges:
            dot_lines.append(f'  "{source}" -> "{target}";')
        dot_lines.append("}")
        graph = "\n".join(dot_lines)
        
        # Create causal model
        causal_model = CausalModel(
            data=data_for_dowhy,
            treatment='treatment',
            outcome='outcome',
            graph=graph
        )
        
        return causal_model
    
    def test_placebo_treatment(self) -> Dict:
        """
        Placebo Treatment Test
        
        Shuffle treatment randomly - if we still find an effect,
        our original estimate might be spurious
        """
        logger.info("\n1. Running Placebo Treatment Test...")
        logger.info("   (Shuffling treatment assignment, expecting ~0 effect)")
        
        if not REFUTATION_CONFIG["placebo_test"]["enabled"]:
            return {'skipped': True}
        
        try:
            model = self.build_causal_model_if_needed()
            identified_estimand = model.identify_effect()
            
            # Estimate original effect
            original_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            # Refute with placebo
            refutation = model.refute_estimate(
                identified_estimand,
                original_estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=REFUTATION_CONFIG["placebo_test"]["num_simulations"]
            )
            
            # Parse results
            new_effect = refutation.new_effect
            p_value = refutation.refutation_result.get('p_value', None)
            
            # Test passes if placebo effect is close to 0
            passed = abs(new_effect) < abs(original_estimate.value) * 0.3
            
            result = {
                'test': 'placebo_treatment',
                'passed': passed,
                'original_effect': float(original_estimate.value),
                'placebo_effect': float(new_effect),
                'p_value': float(p_value) if p_value else None,
                'interpretation': (
                    'PASS: Placebo effect is small, original estimate robust'
                    if passed else
                    'FAIL: Placebo effect is large, original estimate may be spurious'
                )
            }
            
            logger.info(f"   Original effect: ${result['original_effect']:.2f}")
            logger.info(f"   Placebo effect: ${result['placebo_effect']:.2f}")
            logger.info(f"   ✓ {'PASS' if passed else '✗ FAIL'}")
            
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Placebo test failed: {str(e)}")
            return {
                'test': 'placebo_treatment',
                'passed': False,
                'error': str(e)
            }
    
    def test_random_common_cause(self) -> Dict:
        """
        Random Common Cause Test
        
        Add a random variable - if estimate changes significantly,
        we might be missing important confounders
        """
        logger.info("\n2. Running Random Common Cause Test...")
        logger.info("   (Adding random confounder, checking stability)")
        
        if not REFUTATION_CONFIG["random_common_cause"]["enabled"]:
            return {'skipped': True}
        
        try:
            model = self.build_causal_model_if_needed()
            identified_estimand = model.identify_effect()
            
            # Estimate original effect
            original_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            # Refute with random common cause
            refutation = model.refute_estimate(
                identified_estimand,
                original_estimate,
                method_name="random_common_cause"
            )
            
            new_effect = refutation.new_effect
            
            # Test passes if effect is stable
            change_pct = abs((new_effect - original_estimate.value) / original_estimate.value)
            passed = change_pct < 0.2  # Less than 20% change
            
            result = {
                'test': 'random_common_cause',
                'passed': passed,
                'original_effect': float(original_estimate.value),
                'new_effect': float(new_effect),
                'change_pct': float(change_pct * 100),
                'interpretation': (
                    'PASS: Estimate stable with random confounder'
                    if passed else
                    'FAIL: Estimate changed significantly, might be missing confounders'
                )
            }
            
            logger.info(f"   Original effect: ${result['original_effect']:.2f}")
            logger.info(f"   New effect: ${result['new_effect']:.2f}")
            logger.info(f"   Change: {result['change_pct']:.1f}%")
            logger.info(f"   ✓ {'PASS' if passed else '✗ FAIL'}")
            
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Random common cause test failed: {str(e)}")
            return {
                'test': 'random_common_cause',
                'passed': False,
                'error': str(e)
            }
    
    def test_subset_validation(self) -> Dict:
        """
        Subset Validation Test
        
        Estimate on a random subset - if dramatically different,
        estimate might not be stable
        """
        logger.info("\n3. Running Subset Validation Test...")
        logger.info("   (Estimating on random subset)")
        
        if not REFUTATION_CONFIG["subset_validation"]["enabled"]:
            return {'skipped': True}
        
        try:
            # Split data
            test_fraction = REFUTATION_CONFIG["subset_validation"]["test_fraction"]
            train_df = self.df.sample(frac=1-test_fraction, random_state=42)
            test_df = self.df.drop(train_df.index)
            
            # Estimate on both
            effects = []
            for name, data in [('train', train_df), ('test', test_df)]:
                feature_cols = [col for col in CANONICAL_SCHEMA["feature_columns"] 
                               if col in data.columns]
                data_subset = data[feature_cols + ['treatment', 'outcome']].copy().fillna(0)
                
                # Simple ATE calculation
                treated_outcome = data_subset[data_subset['treatment'] == 1]['outcome'].mean()
                control_outcome = data_subset[data_subset['treatment'] == 0]['outcome'].mean()
                effect = treated_outcome - control_outcome
                effects.append((name, effect))
            
            train_effect = effects[0][1]
            test_effect = effects[1][1]
            
            # Correlation check
            correlation = abs(test_effect - train_effect) / abs(train_effect) if train_effect != 0 else 0
            passed = correlation < 0.3  # Less than 30% difference
            
            result = {
                'test': 'subset_validation',
                'passed': passed,
                'train_effect': float(train_effect),
                'test_effect': float(test_effect),
                'difference_pct': float(correlation * 100),
                'interpretation': (
                    'PASS: Estimate consistent across subsets'
                    if passed else
                    'FAIL: Estimate varies significantly across subsets'
                )
            }
            
            logger.info(f"   Train effect: ${train_effect:.2f}")
            logger.info(f"   Test effect: ${test_effect:.2f}")
            logger.info(f"   Difference: {correlation*100:.1f}%")
            logger.info(f"   ✓ {'PASS' if passed else '✗ FAIL'}")
            
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Subset validation failed: {str(e)}")
            return {
                'test': 'subset_validation',
                'passed': False,
                'error': str(e)
            }
    
    def test_data_subset_refutation(self) -> Dict:
        """
        Data Subset Refutation
        
        Use only a portion of data - check if estimate holds
        """
        logger.info("\n4. Running Data Subset Refutation...")
        logger.info("   (Using random subset of data)")
        
        if not REFUTATION_CONFIG["data_subset"]["enabled"]:
            return {'skipped': True}
        
        try:
            model = self.build_causal_model_if_needed()
            identified_estimand = model.identify_effect()
            
            # Estimate original effect
            original_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            # Refute with data subset
            subset_fraction = REFUTATION_CONFIG["data_subset"]["subset_fraction"]
            refutation = model.refute_estimate(
                identified_estimand,
                original_estimate,
                method_name="data_subset_refuter",
                subset_fraction=subset_fraction
            )
            
            new_effect = refutation.new_effect
            
            # Test passes if effect is similar
            change_pct = abs((new_effect - original_estimate.value) / original_estimate.value)
            passed = change_pct < 0.25
            
            result = {
                'test': 'data_subset',
                'passed': passed,
                'original_effect': float(original_estimate.value),
                'subset_effect': float(new_effect),
                'subset_fraction': float(subset_fraction),
                'change_pct': float(change_pct * 100),
                'interpretation': (
                    'PASS: Estimate consistent on data subset'
                    if passed else
                    'FAIL: Estimate unstable on smaller dataset'
                )
            }
            
            logger.info(f"   Original effect: ${result['original_effect']:.2f}")
            logger.info(f"   Subset effect: ${result['subset_effect']:.2f}")
            logger.info(f"   Change: {result['change_pct']:.1f}%")
            logger.info(f"   ✓ {'PASS' if passed else '✗ FAIL'}")
            
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Data subset refutation failed: {str(e)}")
            return {
                'test': 'data_subset',
                'passed': False,
                'error': str(e)
            }
    
    def test_bootstrap_validation(self) -> Dict:
        """
        Bootstrap Validation
        
        Bootstrap resample and re-estimate - check confidence intervals
        """
        logger.info("\n5. Running Bootstrap Validation...")
        logger.info("   (Bootstrap resampling for CI)")
        
        if not REFUTATION_CONFIG["bootstrap"]["enabled"]:
            return {'skipped': True}
        
        try:
            num_simulations = REFUTATION_CONFIG["bootstrap"]["num_simulations"]
            
            # Simple bootstrap
            effects = []
            for i in range(num_simulations):
                boot_df = self.df.sample(n=len(self.df), replace=True, random_state=i)
                
                treated_outcome = boot_df[boot_df['treatment'] == 1]['outcome'].mean()
                control_outcome = boot_df[boot_df['treatment'] == 0]['outcome'].mean()
                effect = treated_outcome - control_outcome
                effects.append(effect)
            
            effects = np.array(effects)
            
            # Calculate statistics
            mean_effect = effects.mean()
            ci_lower = np.percentile(effects, 2.5)
            ci_upper = np.percentile(effects, 97.5)
            
            # Test passes if CI doesn't include 0 (for positive effects)
            passed = ci_lower > 0 if mean_effect > 0 else ci_upper < 0
            
            result = {
                'test': 'bootstrap',
                'passed': passed,
                'mean_effect': float(mean_effect),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'std': float(effects.std()),
                'interpretation': (
                    'PASS: Confidence interval excludes zero, effect is significant'
                    if passed else
                    'FAIL: Confidence interval includes zero, effect not significant'
                )
            }
            
            logger.info(f"   Mean effect: ${mean_effect:.2f}")
            logger.info(f"   95% CI: [${ci_lower:.2f}, ${ci_upper:.2f}]")
            logger.info(f"   ✓ {'PASS' if passed else '✗ FAIL'}")
            
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Bootstrap validation failed: {str(e)}")
            return {
                'test': 'bootstrap',
                'passed': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict:
        """Run all refutation tests"""
        logger.info("=" * 60)
        logger.info("REFUTATION TESTS")
        logger.info("=" * 60)
        
        # Run each test
        self.results['tests']['placebo'] = self.test_placebo_treatment()
        self.results['tests']['random_cause'] = self.test_random_common_cause()
        self.results['tests']['subset_validation'] = self.test_subset_validation()
        self.results['tests']['data_subset'] = self.test_data_subset_refutation()
        self.results['tests']['bootstrap'] = self.test_bootstrap_validation()
        
        # Summarize
        self._summarize_results()
        
        return self.results
    
    def _summarize_results(self):
        """Summarize all test results"""
        # Count passes/fails
        tests_run = []
        tests_passed = []
        
        for test_name, result in self.results['tests'].items():
            if result.get('skipped'):
                continue
            tests_run.append(test_name)
            if result.get('passed', False):
                tests_passed.append(test_name)
        
        pass_rate = len(tests_passed) / len(tests_run) if tests_run else 0
        self.results['overall_pass'] = pass_rate >= 0.6  # At least 60% pass
        
        self.results['summary'] = {
            'total_tests': len(tests_run),
            'passed': len(tests_passed),
            'failed': len(tests_run) - len(tests_passed),
            'pass_rate': float(pass_rate),
        }
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("REFUTATION TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests run: {len(tests_run)}")
        logger.info(f"Passed: {len(tests_passed)}/{len(tests_run)}")
        logger.info(f"Pass rate: {pass_rate:.0%}")
        
        if self.results['overall_pass']:
            logger.info("✓ OVERALL: PASS - Causal estimates are robust")
        else:
            logger.warning("⚠ OVERALL: CAUTION - Some refutation tests failed")
        
        logger.info("=" * 60)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load canonical data"""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    logger.info(f"✓ Loaded {len(df)} records")
    return df


def save_report(results: Dict, output_path: Path):
    """Save refutation report"""
    logger.info(f"\nSaving refutation report to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Report saved")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("STEP 5: REFUTATION TESTS")
    logger.info("=" * 60)
    
    # Load data
    df = load_data(FILE_PATHS["canonical_data"])
    
    # Run refutation tests
    tester = RefutationTester(df)
    results = tester.run_all_tests()
    
    # Save report
    save_report(results, FILE_PATHS["refutation_report"])
    
    return results['overall_pass']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)