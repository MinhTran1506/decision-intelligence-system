# ============================================================
# Decision Intelligence System - Airflow Setup Script
# ============================================================
# 
# IMPORTANT: Apache Airflow does not officially support Windows.
# Use one of these options:
#
# Option 1: Docker (Recommended)
# Option 2: WSL2 (Windows Subsystem for Linux)
# Option 3: Run pipeline directly without Airflow
# ============================================================

Write-Host "============================================================"
Write-Host "  Decision Intelligence System - Airflow Setup"
Write-Host "============================================================"
Write-Host ""

# Check if Docker is available
$dockerAvailable = $null -ne (Get-Command docker -ErrorAction SilentlyContinue)

if ($dockerAvailable) {
    Write-Host "✅ Docker detected! Using Docker Compose for Airflow."
    Write-Host ""
    Write-Host "Starting Airflow with Docker Compose..."
    Write-Host "Run: docker-compose -f docker-compose-airflow.yml up -d"
    Write-Host ""
} else {
    Write-Host "⚠️  Docker not detected."
    Write-Host ""
    Write-Host "Options to run Airflow:"
    Write-Host ""
    Write-Host "1. Install Docker Desktop and run:"
    Write-Host "   docker-compose -f docker-compose-airflow.yml up -d"
    Write-Host ""
    Write-Host "2. Use WSL2 (Windows Subsystem for Linux):"
    Write-Host "   wsl"
    Write-Host "   pip install apache-airflow"
    Write-Host "   airflow standalone"
    Write-Host ""
    Write-Host "3. Run pipeline directly (no Airflow):"
    Write-Host "   python run_pipeline.py"
    Write-Host ""
}

Write-Host "============================================================"
Write-Host "  Alternative: Run Pipeline Directly"
Write-Host "============================================================"
Write-Host ""
Write-Host "You can run the same pipeline without Airflow:"
Write-Host ""
Write-Host "  python run_pipeline.py"
Write-Host ""
Write-Host "This runs all steps: data generation, quality checks,"
Write-Host "causal estimation, refutation tests, and model registration."
Write-Host ""
