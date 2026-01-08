"""
Enhanced FastAPI with Real-time WebSocket Support

Additional endpoints:
- WebSocket for live updates
- Model versioning and comparison
- A/B test result tracking
- Feature importance analysis
- Batch scoring
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import asyncio
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import API_CONFIG, FILE_PATHS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global state
uplift_data = None
model_metadata = {}
connected_clients = []


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup, cleanup on shutdown"""
    global uplift_data, model_metadata
    
    # Startup
    logger.info("Loading model data...")
    
    try:
        uplift_data = pd.read_parquet(FILE_PATHS["uplift_scores"])
        logger.info(f"[OK] Loaded {len(uplift_data)} uplift scores")
        
        model_metadata = {
            "model_version": "v1.0",
            "loaded_at": datetime.now().isoformat(),
            "n_records": len(uplift_data),
        }
        
        # Start background task for simulating real-time events
        asyncio.create_task(simulate_realtime_events())
        
        logger.info("[OK] API ready with real-time capabilities")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down API...")


async def simulate_realtime_events():
    """Background task to simulate real-time event generation"""
    await asyncio.sleep(5)  # Wait for startup
    
    while True:
        try:
            # Simulate new event
            event = {
                'type': 'new_score',
                'timestamp': datetime.now().isoformat(),
                'user_id': f'user_{np.random.randint(1000000, 9999999)}',
                'uplift_score': float(np.random.normal(45, 20)),
                'segment': np.random.choice(['High Uplift', 'Medium-High', 'Medium-Low', 'Low Uplift']),
                'recommendation': np.random.choice(['TREAT', 'NO_TREATMENT'], p=[0.3, 0.7]),
            }
            
            # Broadcast to all connected WebSocket clients
            await manager.broadcast(event)
            
            # Random interval between 2-5 seconds
            await asyncio.sleep(np.random.uniform(2, 5))
            
        except Exception as e:
            logger.error(f"Error in real-time simulation: {e}")
            await asyncio.sleep(5)


# Initialize FastAPI app
app = FastAPI(
    title="Decision Intelligence Studio - Enhanced API",
    description="Real-time causal inference API with WebSocket support",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS
if API_CONFIG["enable_cors"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Enhanced Pydantic models
class BatchScoreRequest(BaseModel):
    """Batch scoring request"""
    users: List[Dict[str, Any]] = Field(..., description="List of user feature dicts")


class ABTestResult(BaseModel):
    """A/B test result submission"""
    test_id: str
    segment: str
    predicted_uplift: float
    observed_uplift: float
    sample_size: int
    confidence_level: float
    start_date: str
    end_date: str


class ModelVersion(BaseModel):
    """Model version information"""
    version: str
    training_date: str
    ate_estimate: float
    cate_std: float
    refutation_pass_rate: float
    test_mae: Optional[float] = None


class FeatureImportanceResponse(BaseModel):
    """Feature importance"""
    feature_name: str
    importance: float
    rank: int


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates
    
    Clients receive:
    - New uplift scores as they're computed
    - System statistics updates
    - Alert notifications
    """
    await manager.connect(websocket)
    
    try:
        # Send initial stats
        await websocket.send_json({
            'type': 'connection_established',
            'timestamp': datetime.now().isoformat(),
            'message': 'Connected to Decision Intelligence Studio real-time feed'
        })
        
        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            
            # Echo back or handle client requests
            if data == 'ping':
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    """Enhanced health check with system info"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": uplift_data is not None,
        "active_websocket_connections": len(manager.active_connections),
        "version": "2.0.0",
    }


@app.post("/batch-score")
async def batch_score(request: BatchScoreRequest):
    """
    Score multiple users in batch
    
    More efficient than calling /score multiple times
    """
    if uplift_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Simulate batch scoring (in production, use actual model)
    results = []
    for user in request.users:
        score = float(np.random.normal(45, 20))
        results.append({
            'user_id': user.get('user_id', 'unknown'),
            'uplift_score': score,
            'segment': 'High Uplift' if score > 60 else 'Medium' if score > 30 else 'Low',
            'recommendation': 'TREAT' if score > 40 else 'NO_TREATMENT',
        })
    
    return {
        'batch_size': len(results),
        'results': results,
        'processing_time_ms': len(results) * 5,  # Simulate processing time
    }


@app.get("/feature-importance")
async def get_feature_importance() -> List[FeatureImportanceResponse]:
    """
    Get feature importance for heterogeneous treatment effects
    
    Shows which features drive the most variation in uplift
    """
    # Simulate feature importance (in production, extract from model)
    importance_data = [
        {'feature_name': 'engagement_score', 'importance': 0.35},
        {'feature_name': 'past_purchases', 'importance': 0.28},
        {'feature_name': 'income_level', 'importance': 0.18},
        {'feature_name': 'age', 'importance': 0.12},
        {'feature_name': 'days_since_signup', 'importance': 0.07},
    ]
    
    # Sort and add ranks
    importance_data = sorted(importance_data, key=lambda x: x['importance'], reverse=True)
    for i, item in enumerate(importance_data):
        item['rank'] = i + 1
    
    return [FeatureImportanceResponse(**item) for item in importance_data]


@app.post("/ab-test/submit")
async def submit_ab_test_result(result: ABTestResult):
    """
    Submit A/B test results for model validation
    
    This creates a feedback loop for continuous model improvement
    """
    # Store result (in production, write to database)
    logger.info(f"A/B test result received: {result.test_id}")
    
    # Calculate prediction error
    error = abs(result.predicted_uplift - result.observed_uplift)
    error_pct = (error / result.predicted_uplift * 100) if result.predicted_uplift else 0
    
    # Determine if calibration is good
    well_calibrated = error_pct < 10  # Within 10% is good
    
    return {
        'test_id': result.test_id,
        'status': 'accepted',
        'prediction_error': round(error, 2),
        'error_percentage': round(error_pct, 1),
        'well_calibrated': well_calibrated,
        'message': 'Test result recorded successfully. Model will be updated in next training cycle.' if well_calibrated else 'Large prediction error detected. Consider model retraining.',
    }


@app.get("/ab-test/history")
async def get_ab_test_history():
    """Get historical A/B test results"""
    # Simulate historical results
    history = [
        {
            'test_id': 'TEST-001',
            'segment': 'High Uplift',
            'predicted_uplift': 89.4,
            'observed_uplift': 87.6,
            'sample_size': 1000,
            'error_pct': 2.0,
            'status': 'Completed',
            'date': '2024-12-01',
        },
        {
            'test_id': 'TEST-002',
            'segment': 'Medium-High',
            'predicted_uplift': 52.2,
            'observed_uplift': 49.8,
            'sample_size': 1500,
            'error_pct': 4.6,
            'status': 'Completed',
            'date': '2024-12-15',
        },
        {
            'test_id': 'TEST-003',
            'segment': 'All Users',
            'predicted_uplift': 45.2,
            'observed_uplift': None,
            'sample_size': 5000,
            'error_pct': None,
            'status': 'Running',
            'date': '2025-01-01',
        },
    ]
    
    return {
        'total_tests': len(history),
        'completed': sum(1 for t in history if t['status'] == 'Completed'),
        'avg_error_pct': np.mean([t['error_pct'] for t in history if t['error_pct'] is not None]),
        'tests': history,
    }


@app.get("/models/list")
async def list_models() -> List[ModelVersion]:
    """List all model versions"""
    models = [
        {
            'version': 'v1.0',
            'training_date': '2025-01-08',
            'ate_estimate': 45.23,
            'cate_std': 18.45,
            'refutation_pass_rate': 100.0,
            'test_mae': 2.34,
        },
        {
            'version': 'v0.9',
            'training_date': '2024-12-15',
            'ate_estimate': 43.87,
            'cate_std': 16.23,
            'refutation_pass_rate': 100.0,
            'test_mae': 3.45,
        },
    ]
    
    return [ModelVersion(**m) for m in models]


@app.post("/models/compare")
async def compare_models(version1: str, version2: str):
    """Compare two model versions"""
    # Simulate comparison
    return {
        'version1': version1,
        'version2': version2,
        'comparison': {
            'ate_difference': 1.36,
            'mae_difference': -1.11,
            'recommendation': f'{version1} is better (lower MAE)',
        }
    }


@app.get("/analytics/time-series")
async def get_time_series_analytics():
    """Get time-series analytics of uplift scores"""
    # Simulate time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    data = {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'avg_uplift': [float(45 + np.random.normal(0, 5)) for _ in range(30)],
        'treatment_rate': [float(0.3 + np.random.normal(0, 0.05)) for _ in range(30)],
        'high_value_pct': [float(0.25 + np.random.normal(0, 0.03)) for _ in range(30)],
    }
    
    return data


@app.get("/analytics/segment-performance")
async def get_segment_performance():
    """Detailed segment performance metrics"""
    if uplift_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    segment_stats = {}
    for segment in uplift_data['segment_name'].unique():
        seg_data = uplift_data[uplift_data['segment_name'] == segment]
        
        segment_stats[segment] = {
            'count': len(seg_data),
            'mean_uplift': float(seg_data['uplift_score'].mean()),
            'median_uplift': float(seg_data['uplift_score'].median()),
            'std_uplift': float(seg_data['uplift_score'].std()),
            'mean_outcome': float(seg_data['outcome'].mean()),
            'treatment_rate': float(seg_data['treatment'].mean()),
        }
    
    return segment_stats


@app.get("/alerts/active")
async def get_active_alerts():
    """Get active system alerts"""
    alerts = [
        {
            'id': 'alert_001',
            'type': 'LOW_TREATMENT_RATE',
            'severity': 'WARNING',
            'message': 'Treatment rate below threshold (4.8% vs 10% target)',
            'timestamp': (datetime.now() - timedelta(minutes=2)).isoformat(),
        },
        {
            'id': 'alert_002',
            'type': 'HIGH_VALUE_CUSTOMER',
            'severity': 'INFO',
            'message': 'High-value customer detected (user_00234567, uplift: $89.45)',
            'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
        }
    ]
    
    return {
        'total': len(alerts),
        'by_severity': {
            'WARNING': 1,
            'INFO': 1,
        },
        'alerts': alerts,
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.realtime_api:app",
        host=API_CONFIG["host"],
        port=8001,  # Different port for enhanced API
        reload=True,
        log_level="info"
    )