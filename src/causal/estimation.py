"""
Causal Estimation Pipeline using DoWhy and EconML

This module implements:
1. Causal graph construction and identification (DoWhy)
2. Average Treatment Effect (ATE) estimation
3. Conditional Average Treatment Effect (CATE) estimation using CausalForestDML
4. Uplift score generation for each individual
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from joblib import dump
from datetime import datetime

# Causal inference libraries
from dowhy import CausalModel
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    MODEL_CONFIG,
    CAUSAL_GRAPH,
    CANONICAL_SCHEMA,
    FILE_PATHS,
    SEGMENTATION,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CausalEstimator:
    """
    End-to-end causal estimation pipeline
    
    Uses:
    - DoWhy for causal identification
    - EconML CausalForestDML for heterogeneous effect estimation
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.causal_model = None
        self.identified_estimand = None
        self.cate_estimator = None
        self.results = {}
        
    def prepare_data(self) -> tuple:
        """Prepare data for causal estimation"""
        logger.info("Preparing data for causal estimation...")
        
        # Get feature columns
        feature_cols = [col for col in CANONICAL_SCHEMA["feature_columns"] 
                       if col in self.df.columns]
        
        # Handle missing values
        X = self.df[feature_cols].fillna(0)
        T = self.df['treatment'].values
        Y = self.df['outcome'].values
        
        logger.info(f"  Features: {len(feature_cols)} columns")
        logger.info(f"  Samples: {len(X)} records")
        logger.info(f"  Treatment rate: {T.mean():.1%}")
        logger.info(f"  Mean outcome: ${Y.mean():.2f}")
        
        return X, T, Y, feature_cols
    
    def build_causal_graph(self) -> str:
        """Build causal graph in DOT format from config"""
        logger.info("Building causal graph...")
        
        edges = CAUSAL_GRAPH["edges"]
        
        # Create DOT graph
        dot_lines = ["digraph {"]
        for source, target in edges:
            dot_lines.append(f'  "{source}" -> "{target}";')
        dot_lines.append("}")
        
        graph_dot = "\n".join(dot_lines)
        
        logger.info(f"  Graph with {len(CAUSAL_GRAPH['nodes'])} nodes, "
                   f"{len(edges)} edges")
        
        return graph_dot
    
    def identify_causal_effect(self, X: pd.DataFrame, T: np.ndarray, 
                               Y: np.ndarray) -> None:
        """
        Use DoWhy to identify causal effect
        
        This step ensures we're estimating the right quantity and
        validates our causal assumptions
        """
        logger.info("Identifying causal effect with DoWhy...")
        
        # Create dataframe for DoWhy
        data_for_dowhy = X.copy()
        data_for_dowhy['treatment'] = T
        data_for_dowhy['outcome'] = Y
        
        # Build causal graph
        graph = self.build_causal_graph()
        
        # Create causal model
        self.causal_model = CausalModel(
            data=data_for_dowhy,
            treatment='treatment',
            outcome='outcome',
            graph=graph
        )
        
        # Identify estimand
        self.identified_estimand = self.causal_model.identify_effect(
            proceed_when_unidentifiable=False
        )
        
        logger.info("  ✓ Causal effect identified")
        logger.info(f"\n{self.identified_estimand}")
        
    def estimate_ate_dowhy(self) -> dict:
        """Estimate Average Treatment Effect using DoWhy"""
        logger.info("Estimating ATE with DoWhy (backdoor adjustment)...")
        
        # Use linear regression with backdoor adjustment
        estimate = self.causal_model.estimate_effect(
            self.identified_estimand,
            method_name="backdoor.linear_regression",
            method_params={
                "need_conditional_estimates": False
            }
        )
        
        ate_value = estimate.value
        
        logger.info(f"  ✓ ATE (DoWhy): ${ate_value:.2f}")
        
        return {
            'method': 'dowhy_backdoor',
            'ate': float(ate_value),
            'estimand': str(self.identified_estimand)
        }
    
    def estimate_cate_econml(self, X: pd.DataFrame, T: np.ndarray, 
                             Y: np.ndarray) -> dict:
        """
        Estimate Conditional Average Treatment Effect using EconML
        
        Uses CausalForestDML which:
        - Handles confounding via double machine learning
        - Estimates heterogeneous effects
        - Provides individual-level uplift scores
        """
        logger.info("Estimating CATE with EconML CausalForestDML...")
        
        # Get model config
        cf_config = MODEL_CONFIG["causal_forest"]
        base_config = MODEL_CONFIG["base_models"]
        
        # Create base models
        # Use regressors for both Y and T models (works for continuous and discrete)
        model_y = RandomForestRegressor(**base_config["model_y"])
        model_t = RandomForestRegressor(**base_config["model_t"])
        
        # Create CATE estimator
        self.cate_estimator = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=cf_config["n_estimators"],
            min_samples_leaf=cf_config["min_samples_leaf"],
            max_depth=cf_config["max_depth"],
            random_state=cf_config["random_state"],
            n_jobs=cf_config["n_jobs"],
            verbose=0
        )
        
        # Fit model
        logger.info("  Training CATE model (this may take a minute)...")
        self.cate_estimator.fit(Y, T, X=X, W=None)
        
        # Get individual treatment effects
        cate_estimates = self.cate_estimator.effect(X)
        
        # Calculate ATE from CATE (should be close to DoWhy ATE)
        ate_from_cate = cate_estimates.mean()
        
        logger.info(f"  ✓ CATE model trained")
        logger.info(f"  ATE (from CATE): ${ate_from_cate:.2f}")
        logger.info(f"  CATE range: ${cate_estimates.min():.2f} to ${cate_estimates.max():.2f}")
        logger.info(f"  CATE std: ${cate_estimates.std():.2f}")
        
        # Get feature importance
        try:
            feature_importance = self.cate_estimator.feature_importances()
            top_features = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head(5)
            
            logger.info("\n  Top 5 features for heterogeneity:")
            for _, row in top_features.iterrows():
                logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        except:
            logger.warning("  Could not extract feature importance")
        
        return {
            'method': 'econml_causal_forest_dml',
            'ate_from_cate': float(ate_from_cate),
            'cate_mean': float(cate_estimates.mean()),
            'cate_std': float(cate_estimates.std()),
            'cate_min': float(cate_estimates.min()),
            'cate_max': float(cate_estimates.max()),
            'cate_estimates': cate_estimates
        }
    
    def compute_confidence_intervals(self, X: pd.DataFrame) -> tuple:
        """Compute confidence intervals for CATE estimates"""
        logger.info("Computing confidence intervals...")
        
        try:
            # Get confidence intervals
            cate_lower, cate_upper = self.cate_estimator.effect_interval(
                X, alpha=0.05  # 95% CI
            )
            
            logger.info("  ✓ Confidence intervals computed")
            return cate_lower, cate_upper
        except Exception as e:
            logger.warning(f"  Could not compute CI: {e}")
            # Return None arrays
            return None, None
    
    def segment_users(self, cate_estimates: np.ndarray) -> np.ndarray:
        """Segment users based on uplift scores"""
        logger.info("Segmenting users by uplift...")
        
        if SEGMENTATION["method"] == "quantile":
            # Quantile-based segmentation
            segments = pd.qcut(
                cate_estimates,
                q=SEGMENTATION["num_segments"],
                labels=range(SEGMENTATION["num_segments"]),
                duplicates='drop'
            )
        else:
            # Threshold-based segmentation
            thresholds = [0, 20, 40, 60, np.inf]
            segments = pd.cut(
                cate_estimates,
                bins=thresholds,
                labels=range(len(thresholds) - 1)
            )
        
        segment_counts = pd.Series(segments).value_counts().sort_index()
        logger.info("  Segment distribution:")
        for seg, count in segment_counts.items():
            seg_name = SEGMENTATION["segment_names"][seg]
            logger.info(f"    {seg_name}: {count} ({count/len(segments):.1%})")
        
        return segments.astype(int)
    
    def run_estimation(self) -> dict:
        """Run complete causal estimation pipeline"""
        logger.info("\n" + "=" * 60)
        logger.info("CAUSAL ESTIMATION PIPELINE")
        logger.info("=" * 60)
        
        # 1. Prepare data
        X, T, Y, feature_cols = self.prepare_data()
        
        # 2. Identify causal effect
        self.identify_causal_effect(X, T, Y)
        
        # 3. Estimate ATE with DoWhy
        ate_results = self.estimate_ate_dowhy()
        
        # 4. Estimate CATE with EconML
        cate_results = self.estimate_cate_econml(X, T, Y)
        
        # 5. Compute confidence intervals
        cate_lower, cate_upper = self.compute_confidence_intervals(X)
        
        # 6. Segment users
        segments = self.segment_users(cate_results['cate_estimates'])
        
        # 7. Compile results
        self.results = {
            'ate': ate_results,
            'cate': cate_results,
            'feature_columns': feature_cols,
            'n_samples': len(X),
            'treatment_rate': float(T.mean()),
            'timestamp': datetime.now().isoformat(),
        }
        
        # 8. Create uplift dataframe
        uplift_df = pd.DataFrame({
            'user_id': self.df['user_id'].values,
            'uplift_score': cate_results['cate_estimates'],
            'segment': segments,
            'segment_name': [SEGMENTATION["segment_names"][s] for s in segments],
            'treatment': T,
            'outcome': Y,
        })
        
        if cate_lower is not None:
            uplift_df['uplift_ci_lower'] = cate_lower
            uplift_df['uplift_ci_upper'] = cate_upper
        
        self.results['uplift_df'] = uplift_df
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("ESTIMATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"ATE (DoWhy): ${ate_results['ate']:.2f}")
        logger.info(f"ATE (EconML): ${cate_results['ate_from_cate']:.2f}")
        logger.info(f"CATE range: ${cate_results['cate_min']:.2f} to ${cate_results['cate_max']:.2f}")
        logger.info("=" * 60)
        
        return self.results


def load_data(file_path: Path) -> pd.DataFrame:
    """Load canonical data"""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    logger.info(f"✓ Loaded {len(df)} records")
    return df


def save_model(estimator: CausalForestDML, output_path: Path):
    """Save trained CATE model"""
    logger.info(f"Saving model to {output_path}...")
    dump(estimator, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Model saved ({size_mb:.2f} MB)")


def save_uplift_scores(uplift_df: pd.DataFrame, output_path: Path):
    """Save uplift scores"""
    logger.info(f"Saving uplift scores to {output_path}...")
    uplift_df.to_parquet(output_path, index=False, compression='snappy')
    logger.info(f"✓ Uplift scores saved ({len(uplift_df)} records)")


def print_segment_analysis(uplift_df: pd.DataFrame):
    """Print segment-level analysis"""
    logger.info("\nSEGMENT ANALYSIS:")
    logger.info("=" * 60)
    
    for seg_name in SEGMENTATION["segment_names"]:
        seg_data = uplift_df[uplift_df['segment_name'] == seg_name]
        if len(seg_data) == 0:
            continue
        
        logger.info(f"\n{seg_name}:")
        logger.info(f"  Count: {len(seg_data)} ({len(seg_data)/len(uplift_df):.1%})")
        logger.info(f"  Mean uplift: ${seg_data['uplift_score'].mean():.2f}")
        logger.info(f"  Median uplift: ${seg_data['uplift_score'].median():.2f}")
        logger.info(f"  Treatment rate: {seg_data['treatment'].mean():.1%}")
        logger.info(f"  Mean outcome: ${seg_data['outcome'].mean():.2f}")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("STEP 4: CAUSAL ESTIMATION")
    logger.info("=" * 60)
    
    # Load data
    df = load_data(FILE_PATHS["canonical_data"])
    
    # Run estimation
    estimator = CausalEstimator(df)
    results = estimator.run_estimation()
    
    # Save model
    save_model(estimator.cate_estimator, FILE_PATHS["model_artifact"])
    
    # Save uplift scores
    save_uplift_scores(results['uplift_df'], FILE_PATHS["uplift_scores"])
    
    # Print segment analysis
    print_segment_analysis(results['uplift_df'])
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Causal estimation complete!")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)