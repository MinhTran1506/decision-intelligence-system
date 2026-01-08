"""
Model Registry Service

Simple SQLite-based model versioning with metadata tracking.
Tracks model versions, training data snapshots, performance metrics,
and enables rollback to previous versions.
"""
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import PROJECT_ROOT, MODELS_DIR, FILE_PATHS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "model_registry.db"


class ModelRegistry:
    """SQLite-based model registry for versioning and tracking"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the model registry database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT DEFAULT 'staged',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    promoted_at TIMESTAMP,
                    deprecated_at TIMESTAMP,
                    
                    -- Training info
                    training_data_path TEXT,
                    training_data_hash TEXT,
                    training_rows INTEGER,
                    training_date TIMESTAMP,
                    
                    -- Git info
                    git_commit TEXT,
                    git_branch TEXT,
                    docker_image TEXT,
                    
                    -- Performance metrics
                    ate_estimate REAL,
                    cate_std REAL,
                    refutation_pass_rate REAL,
                    
                    -- Artifacts
                    model_path TEXT,
                    uplift_scores_path TEXT,
                    
                    -- Metadata
                    config_json TEXT,
                    notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployed_by TEXT,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_status 
                ON model_versions(status)
            """)
            
            conn.commit()
        
        logger.info(f"Model registry initialized at {self.db_path}")
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit and branch"""
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
            
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return {'commit': commit, 'branch': branch}
        except:
            return {'commit': 'unknown', 'branch': 'unknown'}
    
    def _hash_file(self, file_path: Path) -> str:
        """Generate hash of a file for data lineage"""
        if not file_path.exists():
            return "file_not_found"
        
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    
    def register_model(
        self,
        model_name: str,
        version: str,
        ate_estimate: float,
        cate_std: float,
        refutation_pass_rate: float,
        training_rows: int,
        config: Dict[str, Any] = None,
        notes: str = None
    ) -> str:
        """
        Register a new model version
        
        Returns:
            model_id: Unique identifier for this model version
        """
        model_id = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        git_info = self._get_git_info()
        
        # Hash training data for lineage
        training_data_hash = self._hash_file(FILE_PATHS["canonical_data"])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_versions (
                    model_id, model_name, version, status,
                    training_data_path, training_data_hash, training_rows, training_date,
                    git_commit, git_branch,
                    ate_estimate, cate_std, refutation_pass_rate,
                    model_path, uplift_scores_path,
                    config_json, notes
                ) VALUES (?, ?, ?, 'staged',
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?)
            """, (
                model_id, model_name, version,
                str(FILE_PATHS["canonical_data"]), training_data_hash, training_rows, datetime.now().isoformat(),
                git_info['commit'], git_info['branch'],
                ate_estimate, cate_std, refutation_pass_rate,
                str(FILE_PATHS["model_artifact"]), str(FILE_PATHS["uplift_scores"]),
                json.dumps(config) if config else None, notes
            ))
            conn.commit()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def promote_model(self, model_id: str, environment: str = 'production') -> bool:
        """Promote a model to production status"""
        with sqlite3.connect(self.db_path) as conn:
            # Demote current production model
            conn.execute("""
                UPDATE model_versions 
                SET status = 'archived', deprecated_at = ?
                WHERE status = 'production'
            """, (datetime.now().isoformat(),))
            
            # Promote new model
            conn.execute("""
                UPDATE model_versions 
                SET status = 'production', promoted_at = ?
                WHERE model_id = ?
            """, (datetime.now().isoformat(), model_id))
            
            # Record deployment
            conn.execute("""
                INSERT INTO model_deployments (model_id, environment, deployed_by)
                VALUES (?, ?, ?)
            """, (model_id, environment, 'system'))
            
            conn.commit()
        
        logger.info(f"Promoted model {model_id} to {environment}")
        return True
    
    def rollback(self, to_model_id: str = None) -> Optional[str]:
        """
        Rollback to a previous model version
        
        Args:
            to_model_id: Specific model to rollback to. If None, rolls back to previous.
            
        Returns:
            model_id of the now-active model, or None if rollback failed
        """
        with sqlite3.connect(self.db_path) as conn:
            if to_model_id is None:
                # Get the most recent archived model
                cursor = conn.execute("""
                    SELECT model_id FROM model_versions
                    WHERE status = 'archived'
                    ORDER BY deprecated_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if not row:
                    logger.error("No previous model to rollback to")
                    return None
                to_model_id = row[0]
            
            # Demote current production
            conn.execute("""
                UPDATE model_versions 
                SET status = 'archived', deprecated_at = ?
                WHERE status = 'production'
            """, (datetime.now().isoformat(),))
            
            # Restore target model
            conn.execute("""
                UPDATE model_versions 
                SET status = 'production', promoted_at = ?
                WHERE model_id = ?
            """, (datetime.now().isoformat(), to_model_id))
            
            conn.commit()
        
        logger.info(f"Rolled back to model {to_model_id}")
        return to_model_id
    
    def get_production_model(self) -> Optional[Dict]:
        """Get the current production model"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM model_versions WHERE status = 'production'
            """)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_models(self, status: str = None, limit: int = 20) -> List[Dict]:
        """List all model versions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status:
                cursor = conn.execute("""
                    SELECT * FROM model_versions 
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (status, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM model_versions 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def record_metric(
        self, 
        model_id: str, 
        metric_name: str, 
        metric_value: float,
        source: str = 'pipeline'
    ):
        """Record a performance metric for a model"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_metrics (model_id, metric_name, metric_value, source)
                VALUES (?, ?, ?, ?)
            """, (model_id, metric_name, metric_value, source))
            conn.commit()
    
    def get_model_metrics(self, model_id: str) -> List[Dict]:
        """Get all metrics for a model"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM model_metrics 
                WHERE model_id = ?
                ORDER BY recorded_at DESC
            """, (model_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def compare_models(self, model_id_1: str, model_id_2: str) -> Dict:
        """Compare two model versions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT * FROM model_versions WHERE model_id IN (?, ?)
            """, (model_id_1, model_id_2))
            
            models = {row['model_id']: dict(row) for row in cursor.fetchall()}
        
        if len(models) != 2:
            return {'error': 'One or both models not found'}
        
        m1, m2 = models[model_id_1], models[model_id_2]
        
        return {
            'model_1': model_id_1,
            'model_2': model_id_2,
            'ate_diff': (m2.get('ate_estimate') or 0) - (m1.get('ate_estimate') or 0),
            'cate_std_diff': (m2.get('cate_std') or 0) - (m1.get('cate_std') or 0),
            'refutation_diff': (m2.get('refutation_pass_rate') or 0) - (m1.get('refutation_pass_rate') or 0),
            'training_rows_diff': (m2.get('training_rows') or 0) - (m1.get('training_rows') or 0),
        }


# Singleton instance
_registry = None

def get_registry() -> ModelRegistry:
    """Get singleton model registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
