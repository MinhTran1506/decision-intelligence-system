"""
Unit tests for Decision Intelligence Studio services
"""
import pytest
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelRegistry:
    """Tests for model registry service"""
    
    def test_registry_initialization(self):
        """Test registry creates database"""
        from src.services.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_registry.db"
            registry = ModelRegistry(db_path)
            assert db_path.exists()
    
    def test_register_model(self):
        """Test model registration"""
        from src.services.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_registry.db"
            registry = ModelRegistry(db_path)
            
            model_id = registry.register_model(
                model_name="test_model",
                version="v1.0",
                ate_estimate=45.0,
                cate_std=15.0,
                refutation_pass_rate=80.0,
                training_rows=1000
            )
            
            assert model_id is not None
            assert "test_model" in model_id
    
    def test_promote_model(self):
        """Test model promotion"""
        from src.services.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_registry.db"
            registry = ModelRegistry(db_path)
            
            model_id = registry.register_model(
                model_name="test_model",
                version="v1.0",
                ate_estimate=45.0,
                cate_std=15.0,
                refutation_pass_rate=80.0,
                training_rows=1000
            )
            
            success = registry.promote_model(model_id)
            assert success
            
            prod_model = registry.get_production_model()
            assert prod_model is not None
            assert prod_model['model_id'] == model_id
    
    def test_rollback(self):
        """Test model rollback"""
        from src.services.model_registry import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_registry.db"
            registry = ModelRegistry(db_path)
            
            # Register and promote first model
            model_1 = registry.register_model(
                model_name="test_model", version="v1.0",
                ate_estimate=40.0, cate_std=15.0,
                refutation_pass_rate=75.0, training_rows=1000
            )
            registry.promote_model(model_1)
            
            # Register and promote second model
            model_2 = registry.register_model(
                model_name="test_model", version="v2.0",
                ate_estimate=45.0, cate_std=12.0,
                refutation_pass_rate=85.0, training_rows=1500
            )
            registry.promote_model(model_2)
            
            # Rollback
            rolled_back = registry.rollback()
            assert rolled_back == model_1
            
            prod_model = registry.get_production_model()
            assert prod_model['model_id'] == model_1


class TestDriftDetection:
    """Tests for drift detection service"""
    
    def test_baseline_computation(self):
        """Test baseline stats computation"""
        from src.services.drift_detection import DriftDetector
        
        # Create sample data
        df = pd.DataFrame({
            'age': np.random.normal(35, 10, 1000),
            'income_level': np.random.normal(50000, 15000, 1000),
            'engagement_score': np.random.uniform(0, 1, 1000),
            'treatment': np.random.binomial(1, 0.3, 1000),
            'outcome': np.random.normal(100, 30, 1000)
        })
        
        detector = DriftDetector()
        stats = detector.compute_baseline_stats(df)
        
        assert 'features' in stats
        assert 'n_records' in stats
        assert stats['n_records'] == 1000
    
    def test_ks_test(self):
        """Test KS test computation"""
        from src.services.drift_detection import DriftDetector
        
        detector = DriftDetector()
        
        # Same distribution should have low KS
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        ks_stat, p_value = detector.ks_test(baseline, current)
        assert ks_stat < 0.1
        
        # Different distribution should have high KS
        current_shifted = np.random.normal(2, 1, 1000)
        ks_stat_shifted, _ = detector.ks_test(baseline, current_shifted)
        assert ks_stat_shifted > 0.3
    
    def test_psi_calculation(self):
        """Test PSI calculation"""
        from src.services.drift_detection import DriftDetector
        
        detector = DriftDetector()
        
        # Same distribution should have low PSI
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        psi = detector.psi(baseline, current)
        assert psi < 0.1
        
        # Different distribution should have high PSI
        current_shifted = np.random.normal(2, 1, 1000)
        psi_shifted = detector.psi(baseline, current_shifted)
        assert psi_shifted > 0.1


class TestNotificationService:
    """Tests for notification service"""
    
    def test_notification_formatting(self):
        """Test Slack message formatting"""
        from src.services.notifications import NotificationService
        
        notifier = NotificationService()
        
        msg = notifier._format_slack_message(
            title="Test Message",
            status="success",
            details={"key1": "value1", "key2": 123}
        )
        
        assert "attachments" in msg
        assert msg["attachments"][0]["title"] == "Test Message"
    
    def test_send_without_webhook(self):
        """Test send returns True when webhook not configured"""
        from src.services.notifications import NotificationService
        
        notifier = NotificationService()
        notifier.slack_webhook = ""
        notifier.enabled = False
        
        # Should not raise, should return True
        result = notifier.send_slack("Test", "success", {})
        assert result == True


class TestDataQuality:
    """Tests for data quality checks"""
    
    def test_required_columns_check(self):
        """Test required columns validation"""
        from src.data.data_quality_checks import DataQualityChecker
        
        # Complete data
        df = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'event_ts': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'treatment': [0, 1],
            'outcome': [10.0, 20.0]
        })
        
        checker = DataQualityChecker(df)
        result = checker.check_required_columns()
        assert result == True
    
    def test_null_rate_check(self):
        """Test null rate validation"""
        from src.data.data_quality_checks import DataQualityChecker
        
        # Data with acceptable nulls
        df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'] * 100,
            'event_ts': pd.to_datetime(['2024-01-01'] * 300),
            'treatment': [0, 1, 0] * 100,
            'outcome': [10.0, 20.0, 15.0] * 100
        })
        
        checker = DataQualityChecker(df)
        result = checker.check_null_rates()
        assert result == True


class TestCausalEstimation:
    """Tests for causal estimation"""
    
    def test_data_preparation(self):
        """Test data preparation for estimation"""
        from src.causal.estimation import CausalEstimator
        
        # Create sample canonical data
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'user_id': [f'u{i}' for i in range(n)],
            'event_ts': pd.to_datetime('2024-01-01'),
            'treatment': np.random.binomial(1, 0.3, n),
            'outcome': np.random.normal(100, 30, n),
            'age': np.random.normal(35, 10, n),
            'income_level': np.random.normal(50000, 15000, n),
            'engagement_score': np.random.uniform(0, 1, n),
            'past_purchases': np.random.poisson(5, n),
            'days_since_signup': np.random.randint(1, 365, n),
            'region_encoded': np.random.randint(0, 5, n),
            'season_encoded': np.random.randint(0, 4, n),
            'day_of_week': np.random.randint(0, 7, n),
        })
        
        estimator = CausalEstimator(df)
        X, T, Y, feature_cols = estimator.prepare_data()
        
        assert len(X) == n
        assert len(T) == n
        assert len(Y) == n
        assert len(feature_cols) > 0


class TestAPIEndpoints:
    """Tests for API endpoints"""
    
    def test_health_endpoint_format(self):
        """Test health endpoint returns expected format"""
        # This would be an integration test with TestClient
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
