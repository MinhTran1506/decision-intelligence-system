"""
Data Store for Decision Intelligence Studio

Provides persistent storage for:
- A/B Test experiments and results
- Real-time event logs
- Model performance metrics
- System configuration

Uses SQLite for simplicity, can be swapped for PostgreSQL/MySQL in production.
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "decision_intel.db"


class DataStore:
    """SQLite-based data store for the application"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # A/B Tests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                segment TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                sample_size INTEGER NOT NULL,
                control_ratio REAL DEFAULT 0.5,
                predicted_uplift REAL,
                observed_uplift REAL,
                confidence_level REAL,
                p_value REAL,
                start_date TEXT,
                end_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                config_json TEXT
            )
        """)
        
        # Test Assignments table (which users are in which group)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                group_name TEXT NOT NULL,
                assigned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES ab_tests(test_id),
                UNIQUE(test_id, user_id)
            )
        """)
        
        # Test Events table (conversions, outcomes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_value REAL,
                event_ts TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT,
                FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
            )
        """)
        
        # Real-time Events log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS realtime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_ts TEXT DEFAULT CURRENT_TIMESTAMP,
                uplift_score REAL,
                segment TEXT,
                recommendation TEXT,
                features_json TEXT,
                processed BOOLEAN DEFAULT 0
            )
        """)
        
        # Model Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT,
                metadata_json TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    # ==================== A/B Tests ====================
    
    def create_ab_test(
        self,
        test_id: str,
        name: str,
        segment: str,
        sample_size: int,
        description: str = "",
        control_ratio: float = 0.5,
        predicted_uplift: Optional[float] = None,
        config: Optional[Dict] = None
    ) -> bool:
        """Create a new A/B test"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO ab_tests 
                (test_id, name, description, segment, sample_size, control_ratio, 
                 predicted_uplift, status, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'draft', ?)
            """, (
                test_id, name, description, segment, sample_size, 
                control_ratio, predicted_uplift, 
                json.dumps(config) if config else None
            ))
            conn.commit()
            logger.info(f"Created A/B test: {test_id}")
            return True
        except sqlite3.IntegrityError:
            logger.error(f"Test {test_id} already exists")
            return False
        finally:
            conn.close()
    
    def start_ab_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE ab_tests 
            SET status = 'running', start_date = ?, updated_at = ?
            WHERE test_id = ? AND status = 'draft'
        """, (datetime.now().isoformat(), datetime.now().isoformat(), test_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def stop_ab_test(self, test_id: str, observed_uplift: float, p_value: float) -> bool:
        """Stop an A/B test and record results"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE ab_tests 
            SET status = 'completed', 
                end_date = ?, 
                observed_uplift = ?,
                p_value = ?,
                updated_at = ?
            WHERE test_id = ? AND status = 'running'
        """, (
            datetime.now().isoformat(), 
            observed_uplift, 
            p_value,
            datetime.now().isoformat(), 
            test_id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def get_ab_test(self, test_id: str) -> Optional[Dict]:
        """Get a single A/B test"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM ab_tests WHERE test_id = ?", (test_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def list_ab_tests(self, status: Optional[str] = None) -> List[Dict]:
        """List all A/B tests"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute(
                "SELECT * FROM ab_tests WHERE status = ? ORDER BY created_at DESC", 
                (status,)
            )
        else:
            cursor.execute("SELECT * FROM ab_tests ORDER BY created_at DESC")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def assign_user_to_test(self, test_id: str, user_id: str, group: str) -> bool:
        """Assign a user to a test group"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO test_assignments (test_id, user_id, group_name)
                VALUES (?, ?, ?)
            """, (test_id, user_id, group))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def get_user_test_assignment(self, test_id: str, user_id: str) -> Optional[str]:
        """Get user's test group assignment"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT group_name FROM test_assignments 
            WHERE test_id = ? AND user_id = ?
        """, (test_id, user_id))
        
        row = cursor.fetchone()
        conn.close()
        
        return row['group_name'] if row else None
    
    def record_test_event(
        self, 
        test_id: str, 
        user_id: str, 
        event_type: str,
        event_value: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Record a test event (conversion, purchase, etc.)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_events (test_id, user_id, event_type, event_value, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            test_id, user_id, event_type, event_value,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def get_test_results(self, test_id: str) -> Dict:
        """Calculate test results"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get assignments by group
        cursor.execute("""
            SELECT group_name, COUNT(*) as count
            FROM test_assignments
            WHERE test_id = ?
            GROUP BY group_name
        """, (test_id,))
        
        groups = {row['group_name']: row['count'] for row in cursor.fetchall()}
        
        # Get conversions by group
        cursor.execute("""
            SELECT a.group_name, 
                   COUNT(DISTINCT e.user_id) as conversions,
                   SUM(e.event_value) as total_value
            FROM test_assignments a
            LEFT JOIN test_events e ON a.test_id = e.test_id AND a.user_id = e.user_id
            WHERE a.test_id = ? AND e.event_type = 'conversion'
            GROUP BY a.group_name
        """, (test_id,))
        
        conversions = {}
        for row in cursor.fetchall():
            conversions[row['group_name']] = {
                'conversions': row['conversions'] or 0,
                'total_value': row['total_value'] or 0
            }
        
        conn.close()
        
        # Calculate metrics
        results = {
            'test_id': test_id,
            'groups': {},
            'lift': None,
            'is_significant': False
        }
        
        for group, count in groups.items():
            conv = conversions.get(group, {'conversions': 0, 'total_value': 0})
            results['groups'][group] = {
                'users': count,
                'conversions': conv['conversions'],
                'conversion_rate': conv['conversions'] / count if count > 0 else 0,
                'total_value': conv['total_value'],
                'avg_value': conv['total_value'] / count if count > 0 else 0
            }
        
        # Calculate lift if both groups exist
        if 'control' in results['groups'] and 'treatment' in results['groups']:
            control_rate = results['groups']['control']['conversion_rate']
            treatment_rate = results['groups']['treatment']['conversion_rate']
            
            if control_rate > 0:
                results['lift'] = (treatment_rate - control_rate) / control_rate * 100
            
            # Simple significance test (would use proper stats in production)
            n_control = results['groups']['control']['users']
            n_treatment = results['groups']['treatment']['users']
            
            if n_control > 30 and n_treatment > 30:
                # Z-test approximation
                p1 = treatment_rate
                p2 = control_rate
                p = (p1 * n_treatment + p2 * n_control) / (n_treatment + n_control)
                if p > 0 and p < 1:
                    se = np.sqrt(p * (1-p) * (1/n_treatment + 1/n_control))
                    z = (p1 - p2) / se if se > 0 else 0
                    results['is_significant'] = abs(z) > 1.96
        
        return results
    
    # ==================== Real-time Events ====================
    
    def log_realtime_event(
        self,
        user_id: str,
        uplift_score: float,
        segment: str,
        recommendation: str,
        features: Optional[Dict] = None
    ) -> int:
        """Log a real-time scoring event"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO realtime_events 
            (user_id, uplift_score, segment, recommendation, features_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id, uplift_score, segment, recommendation,
            json.dumps(features) if features else None
        ))
        
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return event_id
    
    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get recent real-time events"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM realtime_events 
            ORDER BY event_ts DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_event_stats(self, hours: int = 24) -> Dict:
        """Get event statistics for the last N hours"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_events,
                AVG(uplift_score) as avg_uplift,
                MIN(uplift_score) as min_uplift,
                MAX(uplift_score) as max_uplift,
                SUM(CASE WHEN recommendation = 'TREAT' THEN 1 ELSE 0 END) as treat_count,
                SUM(CASE WHEN segment = 'High Uplift' THEN 1 ELSE 0 END) as high_uplift_count
            FROM realtime_events
            WHERE event_ts > ?
        """, (cutoff,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else {}
    
    def get_unprocessed_events(self, limit: int = 1000) -> List[Dict]:
        """Get unprocessed events for batch processing"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM realtime_events 
            WHERE processed = 0 
            ORDER BY event_ts ASC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def mark_events_processed(self, event_ids: List[int]) -> bool:
        """Mark events as processed"""
        if not event_ids:
            return True
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join(['?' for _ in event_ids])
        cursor.execute(f"""
            UPDATE realtime_events 
            SET processed = 1 
            WHERE id IN ({placeholders})
        """, event_ids)
        
        conn.commit()
        conn.close()
        return True
    
    # ==================== Alerts ====================
    
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """Create a new alert"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (alert_type, severity, message, metadata_json)
            VALUES (?, ?, ?, ?)
        """, (
            alert_type, severity, message,
            json.dumps(metadata) if metadata else None
        ))
        
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return alert_id
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM alerts 
            WHERE is_active = 1 
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def resolve_alert(self, alert_id: int) -> bool:
        """Resolve an alert"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts 
            SET is_active = 0, resolved_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), alert_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    # ==================== Model Metrics ====================
    
    def record_metric(self, model_version: str, metric_name: str, metric_value: float):
        """Record a model metric"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_metrics (model_version, metric_name, metric_value)
            VALUES (?, ?, ?)
        """, (model_version, metric_name, metric_value))
        
        conn.commit()
        conn.close()
    
    def get_metric_history(
        self, 
        model_version: str, 
        metric_name: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get metric history"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM model_metrics
            WHERE model_version = ? AND metric_name = ?
            ORDER BY recorded_at DESC
            LIMIT ?
        """, (model_version, metric_name, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


# Singleton instance
_store_instance = None

def get_store() -> DataStore:
    """Get or create DataStore instance"""
    global _store_instance
    if _store_instance is None:
        _store_instance = DataStore()
    return _store_instance
