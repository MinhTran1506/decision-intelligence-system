"""
FastAPI Service for Decision Intelligence Studio

Endpoints:
- GET /health - Health check
- GET /stats - Model and data statistics
- POST /score - Get uplift scores for users
- POST /simulate - Run what-if simulation
- POST /recommend - Get action recommendations
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import API_CONFIG, DECISION_RULES, FILE_PATHS, SEGMENTATION
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global state
uplift_data = None
model_metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global uplift_data, model_metadata
    
    # Startup: Load model data
    logger.info("Loading model data...")
    
    try:
        # Load uplift scores
        uplift_data = pd.read_parquet(FILE_PATHS["uplift_scores"])
        logger.info(f"[OK] Loaded {len(uplift_data)} uplift scores")
        
        # Load model metadata (if available)
        model_metadata = {
            "model_version": "v1.0",
            "loaded_at": datetime.now().isoformat(),
            "n_records": len(uplift_data),
            "segments": SEGMENTATION["segment_names"],
        }
        
        logger.info("[OK] API ready")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    yield
    
    # Shutdown: cleanup if needed
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    lifespan=lifespan,
)

# Add CORS middleware
if API_CONFIG["enable_cors"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Pydantic models for request/response
class ScoreRequest(BaseModel):
    user_ids: Optional[List[str]] = Field(None, description="List of user IDs to score")
    limit: Optional[int] = Field(100, description="Maximum number of results")


class ScoreResponse(BaseModel):
    user_id: str
    uplift_score: float
    segment: str
    treatment: int
    outcome: float
    recommendation: str


class SimulateRequest(BaseModel):
    segment_filter: Optional[str] = Field(None, description="Segment to target (e.g., 'High Uplift')")
    treatment_change: int = Field(1, description="Treatment change (0->1 or 1->0)")
    sample_size: Optional[int] = Field(None, description="Number of users to include")


class SimulateResponse(BaseModel):
    scenario: str
    n_users: int
    predicted_incremental_revenue: float
    cost: float
    roi: float
    confidence_interval: Dict[str, float]


class RecommendRequest(BaseModel):
    budget: Optional[float] = Field(None, description="Campaign budget")
    min_roi: Optional[float] = Field(None, description="Minimum ROI threshold")


class RecommendResponse(BaseModel):
    action: str
    n_users: int
    user_ids: List[str]
    expected_revenue: float
    cost: float
    roi: float
    reason: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": uplift_data is not None,
    }


@app.get("/stats")
async def get_stats():
    """Get model and data statistics"""
    if uplift_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    stats = {
        "model_metadata": model_metadata,
        "data_stats": {
            "total_users": len(uplift_data),
            "treatment_rate": float(uplift_data['treatment'].mean()),
            "mean_uplift": float(uplift_data['uplift_score'].mean()),
            "mean_outcome": float(uplift_data['outcome'].mean()),
        },
        "segments": {}
    }
    
    # Segment-level stats
    for seg_name in SEGMENTATION["segment_names"]:
        seg_data = uplift_data[uplift_data['segment_name'] == seg_name]
        if len(seg_data) > 0:
            stats["segments"][seg_name] = {
                "count": len(seg_data),
                "percentage": float(len(seg_data) / len(uplift_data)),
                "mean_uplift": float(seg_data['uplift_score'].mean()),
                "mean_outcome": float(seg_data['outcome'].mean()),
            }
    
    return stats


@app.post("/score", response_model=List[ScoreResponse])
async def score_users(request: ScoreRequest):
    """
    Get uplift scores and recommendations for users
    
    Returns uplift scores, segments, and action recommendations
    """
    if uplift_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Filter by user IDs if provided
    if request.user_ids:
        results = uplift_data[uplift_data['user_id'].isin(request.user_ids)]
    else:
        results = uplift_data.copy()
    
    # Apply limit
    results = results.head(request.limit)
    
    if len(results) == 0:
        raise HTTPException(status_code=404, detail="No users found")
    
    # Generate recommendations
    responses = []
    for _, row in results.iterrows():
        # Decision logic
        expected_gain = row['uplift_score'] * DECISION_RULES["expected_value_per_conversion"]
        cost = DECISION_RULES["cost_per_treatment"]
        net_benefit = expected_gain - cost
        
        if net_benefit > DECISION_RULES["min_uplift_threshold"]:
            recommendation = "TREAT"
        else:
            recommendation = "NO_TREATMENT"
        
        responses.append(ScoreResponse(
            user_id=row['user_id'],
            uplift_score=float(row['uplift_score']),
            segment=row['segment_name'],
            treatment=int(row['treatment']),
            outcome=float(row['outcome']),
            recommendation=recommendation
        ))
    
    return responses


@app.post("/simulate", response_model=SimulateResponse)
async def simulate_intervention(request: SimulateRequest):
    """
    Run what-if simulation for a segment
    
    Calculates expected incremental revenue, cost, and ROI
    """
    if uplift_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Filter by segment if specified
    if request.segment_filter:
        simulation_data = uplift_data[
            uplift_data['segment_name'] == request.segment_filter
        ]
        if len(simulation_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Segment '{request.segment_filter}' not found"
            )
    else:
        simulation_data = uplift_data.copy()
    
    # Apply sample size limit
    if request.sample_size:
        simulation_data = simulation_data.head(request.sample_size)
    
    # Calculate metrics
    n_users = len(simulation_data)
    
    # Predicted incremental revenue
    predicted_incremental = (
        simulation_data['uplift_score'] * 
        DECISION_RULES["expected_value_per_conversion"]
    ).sum()
    
    # Calculate cost
    cost = n_users * DECISION_RULES["cost_per_treatment"]
    
    # Calculate ROI
    roi = (predicted_incremental - cost) / cost if cost > 0 else 0
    
    # Bootstrap confidence interval
    n_bootstrap = 100
    bootstrap_revenues = []
    for _ in range(n_bootstrap):
        boot_sample = simulation_data.sample(n=len(simulation_data), replace=True)
        boot_revenue = (
            boot_sample['uplift_score'] * 
            DECISION_RULES["expected_value_per_conversion"]
        ).sum()
        bootstrap_revenues.append(boot_revenue)
    
    ci_lower = float(np.percentile(bootstrap_revenues, 2.5))
    ci_upper = float(np.percentile(bootstrap_revenues, 97.5))
    
    return SimulateResponse(
        scenario=request.segment_filter or "All Users",
        n_users=n_users,
        predicted_incremental_revenue=float(predicted_incremental),
        cost=float(cost),
        roi=float(roi),
        confidence_interval={
            "lower": ci_lower,
            "upper": ci_upper
        }
    )


@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    Get action recommendations based on uplift scores and business rules
    
    Applies business constraints (budget, ROI) and returns optimal action
    """
    if uplift_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get business rules
    budget = request.budget or DECISION_RULES["budget_limit"]
    min_roi = request.min_roi or DECISION_RULES["min_roi_threshold"]
    cost_per_user = DECISION_RULES["cost_per_treatment"]
    
    # Calculate expected value for each user
    candidate_data = uplift_data.copy()
    candidate_data['expected_gain'] = (
        candidate_data['uplift_score'] * 
        DECISION_RULES["expected_value_per_conversion"]
    )
    candidate_data['net_benefit'] = candidate_data['expected_gain'] - cost_per_user
    candidate_data['roi'] = candidate_data['net_benefit'] / cost_per_user
    
    # Filter candidates
    candidates = candidate_data[
        (candidate_data['net_benefit'] > DECISION_RULES["min_uplift_threshold"]) &
        (candidate_data['roi'] > min_roi)
    ].sort_values('net_benefit', ascending=False)
    
    # Apply budget constraint
    max_users = int(budget / cost_per_user)
    selected = candidates.head(min(max_users, DECISION_RULES["max_users_per_campaign"]))
    
    if len(selected) == 0:
        return RecommendResponse(
            action="NO_ACTION",
            n_users=0,
            user_ids=[],
            expected_revenue=0.0,
            cost=0.0,
            roi=0.0,
            reason="No users meet profitability criteria"
        )
    
    # Calculate totals
    n_users = len(selected)
    expected_revenue = float(selected['expected_gain'].sum())
    total_cost = float(n_users * cost_per_user)
    overall_roi = (expected_revenue - total_cost) / total_cost
    
    return RecommendResponse(
        action="SEND_PROMOTION",
        n_users=n_users,
        user_ids=selected['user_id'].head(100).tolist(),  # Return max 100 IDs
        expected_revenue=expected_revenue,
        cost=total_cost,
        roi=float(overall_roi),
        reason=f"Selected top {n_users} users with positive ROI"
    )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
        log_level="info"
    )