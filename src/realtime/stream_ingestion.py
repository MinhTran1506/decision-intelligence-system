"""
Real-time Data Ingestion Pipeline

Simulates streaming customer events and processes them in real-time:
- Generates events continuously
- Computes features on-the-fly
- Scores with trained model
- Updates dashboards via WebSocket
- Triggers alerts based on thresholds
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from collections import deque
from joblib import load

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import FILE_PATHS, DECISION_RULES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RealtimeEventGenerator:
    """Generates realistic streaming customer events"""
    
    def __init__(self, rate_per_second: float = 2.0):
        self.rate_per_second = rate_per_second
        self.user_counter = 0
        
    def generate_event(self) -> Dict:
        """Generate a single customer event"""
        self.user_counter += 1
        
        # Realistic distributions
        age = np.random.gamma(2, 15) + 18
        age = np.clip(age, 18, 75)
        
        income_level = np.clip(int(2 + (age - 35) / 20 + np.random.normal(0, 0.5)), 1, 5)
        past_purchases = np.random.poisson(2 + income_level * 0.5)
        days_since_signup = np.random.uniform(0, 365)
        engagement_score = np.clip(
            20 + past_purchases * 3 + np.maximum(0, 100 - days_since_signup / 3) + np.random.normal(0, 10),
            0, 100
        )
        
        # Generate event
        event = {
            'user_id': f'user_{self.user_counter:08d}',
            'event_ts': datetime.now().isoformat(),
            'age': float(age.round(0)),
            'income_level': int(income_level),
            'region_encoded': int(np.random.choice([0, 1, 2, 3])),
            'past_purchases': int(past_purchases),
            'days_since_signup': float(days_since_signup.round(0)),
            'engagement_score': float(engagement_score.round(1)),
            'season_encoded': (datetime.now().month - 1) // 3,
            'day_of_week': datetime.now().weekday(),
        }
        
        return event
    
    async def stream_events(self):
        """Continuously generate events"""
        while True:
            yield self.generate_event()
            await asyncio.sleep(1.0 / self.rate_per_second)


class RealtimeScorer:
    """Scores incoming events in real-time"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = [
            'age', 'income_level', 'region_encoded',
            'past_purchases', 'days_since_signup', 'engagement_score',
            'season_encoded', 'day_of_week'
        ]
        self.load_model()
        
        # Statistics tracking
        self.events_processed = 0
        self.high_value_count = 0
        self.avg_uplift = 0.0
        self.scores_buffer = deque(maxlen=100)
        
    def load_model(self):
        """Load trained CATE model"""
        try:
            model_path = FILE_PATHS["model_artifact"]
            if model_path.exists():
                self.model = load(model_path)
                logger.info(f"✓ Model loaded from {model_path}")
            else:
                logger.warning("Model not found, scores will be simulated")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def score_event(self, event: Dict) -> Dict:
        """Score a single event"""
        # Extract features
        X = pd.DataFrame([event])[self.feature_cols]
        
        # Get uplift score
        if self.model:
            try:
                uplift_score = float(self.model.effect(X)[0])
            except Exception as e:
                logger.warning(f"Model scoring failed: {e}, using simulation")
                uplift_score = self._simulate_score(event)
        else:
            uplift_score = self._simulate_score(event)
        
        # Calculate recommendation
        expected_gain = uplift_score * DECISION_RULES["expected_value_per_conversion"]
        cost = DECISION_RULES["cost_per_treatment"]
        roi = (expected_gain - cost) / cost if cost > 0 else 0
        
        recommendation = "TREAT" if expected_gain > cost + DECISION_RULES["min_uplift_threshold"] else "NO_TREATMENT"
        
        # Determine segment
        if uplift_score > 60:
            segment = "High Uplift"
        elif uplift_score > 40:
            segment = "Medium-High"
        elif uplift_score > 20:
            segment = "Medium-Low"
        else:
            segment = "Low Uplift"
        
        # Update statistics
        self.events_processed += 1
        if recommendation == "TREAT":
            self.high_value_count += 1
        
        self.scores_buffer.append(uplift_score)
        self.avg_uplift = np.mean(self.scores_buffer)
        
        # Build result
        result = {
            **event,
            'uplift_score': round(uplift_score, 2),
            'expected_gain': round(expected_gain, 2),
            'roi': round(roi, 3),
            'recommendation': recommendation,
            'segment': segment,
            'scored_at': datetime.now().isoformat(),
        }
        
        return result
    
    def _simulate_score(self, event: Dict) -> float:
        """Simulate uplift score based on features"""
        base_score = 30
        
        # Engagement factor
        engagement_factor = (event['engagement_score'] - 50) * 0.5
        
        # Income factor
        if event['income_level'] == 3:
            income_factor = 20
        elif event['income_level'] >= 4:
            income_factor = -10
        else:
            income_factor = 5
        
        # Loyalty factor
        loyalty_factor = event['past_purchases'] * 2
        
        # Add noise
        noise = np.random.normal(0, 10)
        
        score = base_score + engagement_factor + income_factor + loyalty_factor + noise
        return float(np.clip(score, 0, 150))
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'events_processed': self.events_processed,
            'high_value_count': self.high_value_count,
            'avg_uplift': round(self.avg_uplift, 2),
            'treatment_rate': round(self.high_value_count / max(self.events_processed, 1), 3),
            'timestamp': datetime.now().isoformat(),
        }


class RealtimeAlertManager:
    """Manages real-time alerts based on thresholds"""
    
    def __init__(self):
        self.alert_thresholds = {
            'high_uplift': 80.0,
            'low_conversion_rate': 0.05,
            'data_quality_score': 0.8,
        }
        self.alerts = deque(maxlen=50)
    
    def check_alerts(self, scored_event: Dict, stats: Dict) -> List[Dict]:
        """Check if any alerts should be triggered"""
        new_alerts = []
        
        # High-value customer alert
        if scored_event['uplift_score'] > self.alert_thresholds['high_uplift']:
            alert = {
                'type': 'HIGH_VALUE_CUSTOMER',
                'severity': 'INFO',
                'message': f"High-value customer detected: {scored_event['user_id']} (uplift: ${scored_event['uplift_score']})",
                'user_id': scored_event['user_id'],
                'timestamp': datetime.now().isoformat(),
            }
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        # Low treatment rate alert
        if stats['treatment_rate'] < self.alert_thresholds['low_conversion_rate'] and stats['events_processed'] > 50:
            alert = {
                'type': 'LOW_TREATMENT_RATE',
                'severity': 'WARNING',
                'message': f"Treatment rate below threshold: {stats['treatment_rate']:.1%}",
                'timestamp': datetime.now().isoformat(),
            }
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        return new_alerts
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return list(self.alerts)[-n:]


class RealtimePipeline:
    """Orchestrates real-time data processing"""
    
    def __init__(self, rate_per_second: float = 2.0):
        self.generator = RealtimeEventGenerator(rate_per_second)
        self.scorer = RealtimeScorer()
        self.alert_manager = RealtimeAlertManager()
        self.running = False
        
        # Store recent scored events
        self.recent_events = deque(maxlen=100)
        
    async def start(self):
        """Start the real-time pipeline"""
        logger.info("Starting real-time pipeline...")
        self.running = True
        
        async for event in self.generator.stream_events():
            if not self.running:
                break
            
            # Score event
            scored_event = self.scorer.score_event(event)
            self.recent_events.append(scored_event)
            
            # Check alerts
            stats = self.scorer.get_stats()
            alerts = self.alert_manager.check_alerts(scored_event, stats)
            
            # Log high-value events
            if scored_event['recommendation'] == 'TREAT':
                logger.info(f"⭐ High-value: {scored_event['user_id']} | "
                          f"Uplift: ${scored_event['uplift_score']:.2f} | "
                          f"Segment: {scored_event['segment']}")
            
            # Log alerts
            for alert in alerts:
                if alert['severity'] == 'WARNING':
                    logger.warning(f"🚨 {alert['message']}")
                else:
                    logger.info(f"ℹ️ {alert['message']}")
            
            # Yield for external consumers (WebSocket, etc.)
            yield {
                'event': scored_event,
                'stats': stats,
                'alerts': alerts,
            }
    
    def stop(self):
        """Stop the pipeline"""
        logger.info("Stopping real-time pipeline...")
        self.running = False
    
    def get_recent_events(self, n: int = 20) -> List[Dict]:
        """Get recent scored events"""
        return list(self.recent_events)[-n:]
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return self.scorer.get_stats()


# Standalone execution for testing
async def main():
    """Test the real-time pipeline"""
    logger.info("=" * 60)
    logger.info("REAL-TIME INGESTION PIPELINE - TEST MODE")
    logger.info("=" * 60)
    
    pipeline = RealtimePipeline(rate_per_second=5.0)  # 5 events per second
    
    try:
        count = 0
        async for result in pipeline.start():
            count += 1
            
            if count % 10 == 0:
                stats = result['stats']
                logger.info(f"\n📊 Statistics after {stats['events_processed']} events:")
                logger.info(f"   Avg Uplift: ${stats['avg_uplift']:.2f}")
                logger.info(f"   Treatment Rate: {stats['treatment_rate']:.1%}")
                logger.info(f"   High-Value Count: {stats['high_value_count']}")
            
            if count >= 100:  # Process 100 events then stop
                break
    
    except KeyboardInterrupt:
        logger.info("\nStopping pipeline...")
    
    finally:
        pipeline.stop()
        logger.info(f"\n✓ Processed {count} events")


if __name__ == "__main__":
    asyncio.run(main())