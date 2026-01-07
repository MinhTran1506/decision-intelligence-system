"""
Generate realistic synthetic data for marketing campaign simulation

Creates data with:
- Realistic confounding (age, income, behavior affects both treatment and outcome)
- Heterogeneous treatment effects (effect varies by customer segments)
- Business-realistic distributions
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import DATA_CONFIG, FILE_PATHS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_marketing_data(
    n_samples: int = 10000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic marketing campaign data with causal structure
    
    The data generation follows this causal story:
    1. Customer characteristics (age, income, engagement) determine:
       - Whether they receive treatment (selection bias)
       - Their baseline revenue potential
    2. Treatment (promotional discount) has heterogeneous effects:
       - High effect for engaged, middle-income customers
       - Low/negative effect for very high-income customers (discount not needed)
       - Medium effect for others
    
    Args:
        n_samples: Number of customers to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with customer data
    """
    logger.info(f"Generating {n_samples} synthetic customer records...")
    np.random.seed(random_seed)
    
    # 1. Generate customer characteristics (confounders)
    
    # Age: 18-75, skewed towards 30-50
    age = np.random.gamma(shape=2, scale=15, size=n_samples) + 18
    age = np.clip(age, 18, 75)
    
    # Income level: 1-5 (correlates with age)
    income_noise = np.random.normal(0, 0.5, n_samples)
    income_level = np.clip(
        2 + (age - 35) / 20 + income_noise,
        1, 5
    ).astype(int)
    
    # Region: 0=North, 1=South, 2=East, 3=West
    region = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.25, 0.25, 0.2])
    
    # Past purchases: Poisson distribution, influenced by income and age
    past_purchases = np.random.poisson(
        lam=2 + income_level * 0.5 + (age - 35) / 20,
        size=n_samples
    )
    past_purchases = np.clip(past_purchases, 0, 50)
    
    # Days since signup: uniform 0-365
    days_since_signup = np.random.uniform(0, 365, n_samples)
    
    # Engagement score: 0-100, influenced by past purchases and recency
    engagement_base = 20 + past_purchases * 3 + np.maximum(0, 100 - days_since_signup / 3)
    engagement_score = np.clip(
        engagement_base + np.random.normal(0, 10, n_samples),
        0, 100
    )
    
    # Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
    season = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.25, 0.25])
    
    # Day of week: 0=Monday ... 6=Sunday
    day_of_week = np.random.choice(range(7), size=n_samples)
    
    # 2. Generate treatment assignment (with confounding)
    
    # Treatment propensity depends on customer characteristics
    # Companies tend to target engaged, middle-income customers
    propensity_score = 0.2 + (
        0.1 * (engagement_score / 100) +  # Engaged customers more likely
        0.15 * (income_level == 3) +      # Middle income targeted most
        0.05 * (past_purchases > 5) -     # Loyal customers targeted
        0.1 * (income_level >= 4)         # High income less targeted (don't need discount)
    )
    propensity_score = np.clip(propensity_score, 0.05, 0.95)
    
    treatment = np.random.binomial(n=1, p=propensity_score, size=n_samples)
    
    # 3. Generate outcomes with heterogeneous treatment effects
    
    # Baseline revenue (no treatment effect)
    baseline_revenue = (
        50 +                              # Base
        income_level * 15 +               # Income effect
        past_purchases * 5 +              # Loyalty effect
        engagement_score * 0.3 +          # Engagement effect
        (age - 35) * 0.5 +               # Age effect
        np.random.normal(0, 20, n_samples)  # Noise
    )
    baseline_revenue = np.maximum(baseline_revenue, 0)
    
    # Treatment effect (heterogeneous)
    # Effect is positive for engaged, middle-income customers
    # Effect is near zero or negative for high-income customers
    treatment_effect = (
        30 +                                    # Base treatment effect
        20 * (engagement_score > 60) +          # High for engaged
        15 * (income_level == 3) +              # High for middle income
        -10 * (income_level >= 4) +             # Negative for high income
        -5 * (past_purchases < 2) +             # Lower for non-loyal
        10 * (season == 3) +                    # Higher in fall (holiday prep)
        np.random.normal(0, 15, n_samples)      # Individual variation
    )
    
    # Observed outcome
    outcome = baseline_revenue + treatment * treatment_effect
    outcome = np.maximum(outcome, 0)  # Revenue can't be negative
    
    # 4. Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DATA_CONFIG["date_range_days"])
    
    timestamps = [
        start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        for _ in range(n_samples)
    ]
    
    # 5. Create DataFrame
    df = pd.DataFrame({
        'user_id': [f'user_{i:06d}' for i in range(n_samples)],
        'event_ts': timestamps,
        'treatment': treatment,
        'treatment_val': treatment * 10.0,  # $10 discount amount
        'outcome': outcome,
        'age': age.round(0),
        'income_level': income_level,
        'region': region,
        'region_encoded': region,  # For model
        'past_purchases': past_purchases,
        'days_since_signup': days_since_signup.round(0),
        'engagement_score': engagement_score.round(1),
        'season': season,
        'season_encoded': season,
        'day_of_week': day_of_week,
        'propensity_score': propensity_score.round(4),  # For debugging
        'true_treatment_effect': treatment_effect.round(2),  # For validation
    })
    
    # Add metadata
    df['ingested_at'] = datetime.now()
    df['source_table'] = 'synthetic_marketing_events'
    
    # Sort by timestamp
    df = df.sort_values('event_ts').reset_index(drop=True)
    
    # Log summary statistics
    logger.info(f"✓ Generated {len(df)} records")
    logger.info(f"  Treatment rate: {treatment.mean():.1%}")
    logger.info(f"  Mean outcome (treated): ${outcome[treatment==1].mean():.2f}")
    logger.info(f"  Mean outcome (control): ${outcome[treatment==0].mean():.2f}")
    logger.info(f"  True ATE: ${treatment_effect.mean():.2f}")
    logger.info(f"  Date range: {df['event_ts'].min()} to {df['event_ts'].max()}")
    
    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save data to parquet format"""
    logger.info(f"Saving data to {output_path}...")
    df.to_parquet(output_path, index=False, compression='snappy')
    logger.info(f"✓ Data saved ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("STEP 1: GENERATE SAMPLE DATA")
    logger.info("=" * 60)
    
    # Generate data
    df = generate_marketing_data(
        n_samples=DATA_CONFIG["sample_size"],
        random_seed=DATA_CONFIG["random_seed"]
    )
    
    # Save to raw data directory
    save_data(df, FILE_PATHS["raw_data"])
    
    # Print sample records
    logger.info("\nSample records:")
    print(df[['user_id', 'treatment', 'outcome', 'age', 'income_level', 'engagement_score']].head(10))
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Sample data generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()