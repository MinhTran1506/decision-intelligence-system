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
from src.services.data_store import get_store
from src.services.ab_test_manager import get_ab_manager

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


# ==================== A/B Testing Endpoints ====================

class CreateExperimentRequest(BaseModel):
    """Request to create new A/B test"""
    name: str
    segment: str
    sample_size: int
    description: str = ""
    control_ratio: float = 0.5
    hypothesis: str = ""


class AssignUserRequest(BaseModel):
    """Request to assign user to experiment"""
    test_id: str
    user_id: str


class RecordConversionRequest(BaseModel):
    """Request to record conversion event"""
    test_id: str
    user_id: str
    value: float = 1.0
    event_type: str = "conversion"


@app.post("/ab-test/create")
async def create_experiment(request: CreateExperimentRequest):
    """Create a new A/B test experiment"""
    manager = get_ab_manager()
    result = manager.create_experiment(
        name=request.name,
        segment=request.segment,
        sample_size=request.sample_size,
        description=request.description,
        control_ratio=request.control_ratio,
        hypothesis=request.hypothesis
    )
    return result


@app.post("/ab-test/start/{test_id}")
async def start_experiment(test_id: str):
    """Start an A/B test experiment"""
    manager = get_ab_manager()
    return manager.start_experiment(test_id)


@app.post("/ab-test/stop/{test_id}")
async def stop_experiment(test_id: str):
    """Stop an A/B test and get final results"""
    manager = get_ab_manager()
    return manager.stop_experiment(test_id)


@app.post("/ab-test/assign")
async def assign_user_to_experiment(request: AssignUserRequest):
    """Assign a user to an experiment group"""
    manager = get_ab_manager()
    group = manager.assign_user(request.test_id, request.user_id)
    
    if group:
        return {
            "status": "assigned",
            "test_id": request.test_id,
            "user_id": request.user_id,
            "group": group
        }
    return {"status": "error", "message": "Could not assign user. Test may not be running."}


@app.post("/ab-test/conversion")
async def record_conversion(request: RecordConversionRequest):
    """Record a conversion event"""
    manager = get_ab_manager()
    return manager.record_conversion(
        test_id=request.test_id,
        user_id=request.user_id,
        value=request.value,
        event_type=request.event_type
    )


@app.get("/ab-test/{test_id}/results")
async def get_experiment_results(test_id: str):
    """Get current results for an experiment"""
    manager = get_ab_manager()
    return manager.get_experiment_results(test_id)


@app.get("/ab-test/list")
async def list_experiments(status: Optional[str] = None):
    """List all experiments"""
    manager = get_ab_manager()
    tests = manager.list_experiments(status)
    summary = manager.get_summary_stats()
    
    return {
        "summary": summary,
        "tests": tests
    }


@app.post("/ab-test/submit")
async def submit_ab_test_result(result: ABTestResult):
    """
    Submit A/B test results for model validation (legacy endpoint)
    
    This creates a feedback loop for continuous model improvement
    """
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
        'message': 'Test result recorded successfully.' if well_calibrated else 'Large prediction error detected.',
    }


@app.get("/ab-test/history")
async def get_ab_test_history():
    """Get A/B test history from database"""
    manager = get_ab_manager()
    tests = manager.list_experiments()
    
    # Format for API response
    history = []
    for t in tests:
        history.append({
            'test_id': t['test_id'],
            'name': t['name'],
            'segment': t['segment'],
            'predicted_uplift': t['predicted_uplift'],
            'observed_uplift': t['observed_uplift'],
            'sample_size': t['sample_size'],
            'status': t['status'],
            'start_date': t['start_date'],
            'end_date': t['end_date'],
        })
    
    summary = manager.get_summary_stats()
    
    return {
        'total_tests': summary['total_tests'],
        'completed': summary['completed'],
        'running': summary['running'],
        'avg_prediction_error': summary['avg_prediction_error'],
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


# ==================== Real-time Event Logging ====================

class LogEventRequest(BaseModel):
    """Request to log a real-time event"""
    user_id: str
    uplift_score: float
    segment: str
    recommendation: str
    features: Optional[Dict[str, Any]] = None


@app.post("/events/log")
async def log_realtime_event(request: LogEventRequest):
    """Log a real-time scoring event to the database"""
    store = get_store()
    
    event_id = store.log_realtime_event(
        user_id=request.user_id,
        uplift_score=request.uplift_score,
        segment=request.segment,
        recommendation=request.recommendation,
        features=request.features
    )
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        'type': 'new_score',
        'timestamp': datetime.now().isoformat(),
        'user_id': request.user_id,
        'uplift_score': request.uplift_score,
        'segment': request.segment,
        'recommendation': request.recommendation,
    })
    
    return {
        "status": "logged",
        "event_id": event_id,
        "user_id": request.user_id
    }


@app.get("/events/recent")
async def get_recent_events(limit: int = 100):
    """Get recent real-time events from database"""
    store = get_store()
    events = store.get_recent_events(limit)
    stats = store.get_event_stats(hours=24)
    
    return {
        "events": events,
        "stats_24h": stats
    }


@app.get("/events/stats")
async def get_event_stats(hours: int = 24):
    """Get event statistics for the last N hours"""
    store = get_store()
    return store.get_event_stats(hours)


@app.get("/events/unprocessed")
async def get_unprocessed_events(limit: int = 1000):
    """Get unprocessed events for batch model training"""
    store = get_store()
    events = store.get_unprocessed_events(limit)
    
    return {
        "count": len(events),
        "events": events
    }


@app.post("/events/mark-processed")
async def mark_events_processed(event_ids: List[int]):
    """Mark events as processed after model training"""
    store = get_store()
    success = store.mark_events_processed(event_ids)
    
    return {"status": "success" if success else "failed", "count": len(event_ids)}


# ==================== Alerts ====================

@app.get("/alerts/active")
async def get_active_alerts():
    """Get active system alerts from database"""
    store = get_store()
    db_alerts = store.get_active_alerts()
    
    # Combine with some computed alerts
    alerts = []
    
    for alert in db_alerts:
        alerts.append({
            'id': alert['id'],
            'type': alert['alert_type'],
            'severity': alert['severity'],
            'message': alert['message'],
            'timestamp': alert['created_at'],
        })
    
    # Add computed alerts based on current state
    if uplift_data is not None:
        treatment_rate = uplift_data['treatment'].mean()
        if treatment_rate < 0.1:
            alerts.append({
                'id': 'computed_001',
                'type': 'LOW_TREATMENT_RATE',
                'severity': 'WARNING',
                'message': f'Treatment rate below threshold ({treatment_rate:.1%} vs 10% target)',
                'timestamp': datetime.now().isoformat(),
            })
    
    return {
        'total': len(alerts),
        'by_severity': {
            'WARNING': sum(1 for a in alerts if a['severity'] == 'WARNING'),
            'INFO': sum(1 for a in alerts if a['severity'] == 'INFO'),
            'ERROR': sum(1 for a in alerts if a['severity'] == 'ERROR'),
        },
        'alerts': alerts,
    }


@app.post("/alerts/create")
async def create_alert(alert_type: str, severity: str, message: str):
    """Create a new alert"""
    store = get_store()
    alert_id = store.create_alert(alert_type, severity, message)
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        'type': 'alert',
        'alert_id': alert_id,
        'alert_type': alert_type,
        'severity': severity,
        'message': message,
        'timestamp': datetime.now().isoformat(),
    })
    
    return {"status": "created", "alert_id": alert_id}


@app.post("/alerts/resolve/{alert_id}")
async def resolve_alert(alert_id: int):
    """Resolve an alert"""
    store = get_store()
    success = store.resolve_alert(alert_id)
    
    return {"status": "resolved" if success else "not_found"}


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