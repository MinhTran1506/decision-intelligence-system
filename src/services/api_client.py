"""
API Client Service for Decision Intelligence Studio

Provides a unified interface to interact with all backend APIs.
Used by Streamlit app and other clients.
"""
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DecisionIntelligenceClient:
    """Client for interacting with Decision Intelligence APIs"""
    
    def __init__(
        self, 
        base_api_url: str = "http://localhost:8000",
        enhanced_api_url: str = "http://localhost:8001"
    ):
        self.base_api_url = base_api_url
        self.enhanced_api_url = enhanced_api_url
        self.timeout = 30
        
    def _make_request(
        self, 
        method: str, 
        url: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request with error handling"""
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, timeout=self.timeout)
            elif method.upper() == "DELETE":
                response = requests.delete(url, params=params, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection failed to {url}")
            return {"error": "API not available", "status": "offline"}
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout to {url}")
            return {"error": "Request timeout", "status": "timeout"}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return {"error": str(e), "status": "error"}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": str(e), "status": "error"}

    # ==================== Health & Stats ====================
    
    def check_health(self, enhanced: bool = False) -> Dict:
        """Check API health status"""
        url = f"{self.enhanced_api_url if enhanced else self.base_api_url}/health"
        return self._make_request("GET", url)
    
    def get_stats(self) -> Dict:
        """Get model and data statistics"""
        url = f"{self.base_api_url}/stats"
        return self._make_request("GET", url)
    
    # ==================== Scoring ====================
    
    def score_users(self, user_ids: Optional[List[str]] = None, limit: int = 100) -> Dict:
        """Get uplift scores for users"""
        url = f"{self.base_api_url}/score"
        data = {"user_ids": user_ids, "limit": limit}
        return self._make_request("POST", url, data=data)
    
    def batch_score(self, users: List[Dict]) -> Dict:
        """Score multiple users in batch"""
        url = f"{self.enhanced_api_url}/batch-score"
        data = {"users": users}
        return self._make_request("POST", url, data=data)
    
    def score_single_user(self, user_id: str) -> Dict:
        """Get score for a single user"""
        result = self.score_users(user_ids=[user_id], limit=1)
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return result
    
    # ==================== Simulation ====================
    
    def simulate_intervention(
        self, 
        segment_filter: Optional[str] = None,
        treatment_change: int = 1,
        sample_size: Optional[int] = None
    ) -> Dict:
        """Run what-if simulation"""
        url = f"{self.base_api_url}/simulate"
        data = {
            "segment_filter": segment_filter,
            "treatment_change": treatment_change,
            "sample_size": sample_size
        }
        return self._make_request("POST", url, data=data)
    
    # ==================== Recommendations ====================
    
    def get_recommendations(
        self, 
        budget: Optional[float] = None,
        min_roi: Optional[float] = None
    ) -> Dict:
        """Get action recommendations"""
        url = f"{self.base_api_url}/recommend"
        data = {"budget": budget, "min_roi": min_roi}
        return self._make_request("POST", url, data=data)
    
    # ==================== A/B Testing ====================
    
    def get_ab_test_history(self) -> Dict:
        """Get A/B test history"""
        url = f"{self.enhanced_api_url}/ab-test/history"
        return self._make_request("GET", url)
    
    def submit_ab_test_result(
        self,
        test_id: str,
        segment: str,
        predicted_uplift: float,
        observed_uplift: float,
        sample_size: int,
        confidence_level: float,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Submit A/B test results"""
        url = f"{self.enhanced_api_url}/ab-test/submit"
        data = {
            "test_id": test_id,
            "segment": segment,
            "predicted_uplift": predicted_uplift,
            "observed_uplift": observed_uplift,
            "sample_size": sample_size,
            "confidence_level": confidence_level,
            "start_date": start_date,
            "end_date": end_date
        }
        return self._make_request("POST", url, data=data)
    
    # ==================== Model Management ====================
    
    def list_models(self) -> List[Dict]:
        """List all model versions"""
        url = f"{self.enhanced_api_url}/models/list"
        return self._make_request("GET", url)
    
    def compare_models(self, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        url = f"{self.enhanced_api_url}/models/compare"
        params = {"version1": version1, "version2": version2}
        return self._make_request("POST", url, params=params)
    
    def get_feature_importance(self) -> List[Dict]:
        """Get feature importance"""
        url = f"{self.enhanced_api_url}/feature-importance"
        return self._make_request("GET", url)
    
    # ==================== Analytics ====================
    
    def get_time_series_analytics(self) -> Dict:
        """Get time-series analytics"""
        url = f"{self.enhanced_api_url}/analytics/time-series"
        return self._make_request("GET", url)
    
    def get_segment_performance(self) -> Dict:
        """Get segment performance metrics"""
        url = f"{self.enhanced_api_url}/analytics/segment-performance"
        return self._make_request("GET", url)
    
    # ==================== Alerts ====================
    
    def get_active_alerts(self) -> Dict:
        """Get active system alerts"""
        url = f"{self.enhanced_api_url}/alerts/active"
        return self._make_request("GET", url)


# Singleton instance
_client_instance = None

def get_client() -> DecisionIntelligenceClient:
    """Get or create API client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = DecisionIntelligenceClient()
    return _client_instance
