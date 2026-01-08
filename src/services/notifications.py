"""
Notification Service

Sends alerts and notifications via Slack, email, or other channels.
Used for pipeline completion, failures, and drift alerts.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import MONITORING, PROJECT_ROOT
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class NotificationService:
    """Send notifications via various channels"""
    
    def __init__(self):
        self.slack_webhook = MONITORING.get("slack_webhook_url", "")
        self.enabled = bool(self.slack_webhook)
    
    def _format_slack_message(
        self, 
        title: str, 
        status: str, 
        details: Dict[str, Any],
        color: str = "#36a64f"
    ) -> Dict:
        """Format message for Slack webhook"""
        fields = [
            {"title": k, "value": str(v), "short": True}
            for k, v in details.items()
        ]
        
        return {
            "attachments": [{
                "color": color,
                "title": title,
                "fields": fields,
                "footer": "Decision Intelligence Studio",
                "ts": int(datetime.now().timestamp())
            }]
        }
    
    def send_slack(
        self, 
        title: str, 
        status: str = "success",
        details: Dict[str, Any] = None,
        color: str = None
    ) -> bool:
        """
        Send notification to Slack
        
        Args:
            title: Message title
            status: success, warning, error
            details: Key-value pairs to include
            color: Override color (default based on status)
        """
        if not self.enabled:
            logger.info(f"[NOTIFICATION] {title} - {status}")
            logger.info(f"  Details: {details}")
            return True  # Pretend success when webhook not configured
        
        if color is None:
            color_map = {
                "success": "#36a64f",  # Green
                "warning": "#ff9800",  # Orange
                "error": "#f44336",    # Red
                "info": "#2196f3"      # Blue
            }
            color = color_map.get(status, "#36a64f")
        
        payload = self._format_slack_message(title, status, details or {}, color)
        
        try:
            import requests
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack notification sent: {title}")
                return True
            else:
                logger.error(f"Slack notification failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_pipeline_success(
        self,
        pipeline_name: str,
        ate_estimate: float,
        refutation_pass_rate: float,
        n_records: int,
        duration_seconds: float
    ) -> bool:
        """Send pipeline success notification"""
        details = {
            "Pipeline": pipeline_name,
            "ATE Estimate": f"${ate_estimate:.2f}",
            "Refutation Pass Rate": f"{refutation_pass_rate}%",
            "Records Processed": f"{n_records:,}",
            "Duration": f"{duration_seconds:.1f}s",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.send_slack(
            title=f"✅ Pipeline Complete: {pipeline_name}",
            status="success",
            details=details
        )
    
    def send_pipeline_failure(
        self,
        pipeline_name: str,
        step: str,
        error: str
    ) -> bool:
        """Send pipeline failure notification"""
        details = {
            "Pipeline": pipeline_name,
            "Failed Step": step,
            "Error": error[:200],  # Truncate long errors
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.send_slack(
            title=f"❌ Pipeline Failed: {pipeline_name}",
            status="error",
            details=details
        )
    
    def send_drift_alert(
        self,
        feature: str,
        ks_statistic: float,
        threshold: float,
        baseline_mean: float,
        current_mean: float
    ) -> bool:
        """Send drift detection alert"""
        details = {
            "Feature": feature,
            "KS Statistic": f"{ks_statistic:.4f}",
            "Threshold": f"{threshold:.4f}",
            "Baseline Mean": f"{baseline_mean:.4f}",
            "Current Mean": f"{current_mean:.4f}",
            "Drift Detected": "Yes" if ks_statistic > threshold else "No"
        }
        
        return self.send_slack(
            title=f"⚠️ Drift Alert: {feature}",
            status="warning",
            details=details
        )
    
    def send_model_promotion(
        self,
        model_id: str,
        version: str,
        environment: str
    ) -> bool:
        """Send model promotion notification"""
        details = {
            "Model ID": model_id,
            "Version": version,
            "Environment": environment,
            "Promoted At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.send_slack(
            title=f"🚀 Model Promoted: {version}",
            status="info",
            details=details
        )
    
    def send_data_quality_alert(
        self,
        check_name: str,
        passed: bool,
        details: Dict[str, Any]
    ) -> bool:
        """Send data quality check result"""
        status = "success" if passed else "error"
        emoji = "✅" if passed else "❌"
        
        return self.send_slack(
            title=f"{emoji} Data Quality: {check_name}",
            status=status,
            details=details
        )


# Singleton instance
_notifier = None

def get_notifier() -> NotificationService:
    """Get singleton notification service instance"""
    global _notifier
    if _notifier is None:
        _notifier = NotificationService()
    return _notifier


def notify_pipeline_complete(
    ate_estimate: float,
    refutation_pass_rate: float,
    n_records: int,
    duration_seconds: float
):
    """Convenience function to notify pipeline completion"""
    notifier = get_notifier()
    return notifier.send_pipeline_success(
        pipeline_name="Decision Intelligence Pipeline",
        ate_estimate=ate_estimate,
        refutation_pass_rate=refutation_pass_rate,
        n_records=n_records,
        duration_seconds=duration_seconds
    )


def notify_pipeline_failure(step: str, error: str):
    """Convenience function to notify pipeline failure"""
    notifier = get_notifier()
    return notifier.send_pipeline_failure(
        pipeline_name="Decision Intelligence Pipeline",
        step=step,
        error=error
    )
