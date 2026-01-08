#!/bin/bash

# Decision Intelligence Studio - Complete Demo Launcher
# This script starts all services for the full demo experience

set -e

# Detect OS
detect_os() {
    case "$OSTYPE" in
        msys*|mingw*|cygwin*)
            echo "windows"
            ;;
        linux-gnu*)
            echo "linux"
            ;;
        darwin*)
            echo "mac"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

OS_TYPE=$(detect_os)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║         DECISION INTELLIGENCE STUDIO - DEMO LAUNCHER             ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}➜ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running in WSL or Windows
check_wsl() {
    if [[ "$OS_TYPE" == "windows" ]]; then
        print_success "Running in Windows (Git Bash/MINGW)"
        return 0
    elif grep -qi microsoft /proc/version 2>/dev/null; then
        print_success "Running in WSL"
        return 0
    else
        print_warning "Running on $OS_TYPE. Some features may behave differently."
        return 0
    fi
}

# Check if virtual environment is activated
check_venv() {
    print_info "Activating virtual environment..."
    # Detect OS and use appropriate activation path
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    print_success "Virtual environment activated"
}

# Check if data exists
check_data() {
    if [[ -f "data/outputs/uplift_scores.parquet" ]]; then
        print_success "Pipeline data found"
        return 0
    else
        print_warning "Pipeline data not found"
        return 1
    fi
}

# Run pipeline if needed
run_pipeline() {
    print_info "Running pipeline to generate data..."
    python run_pipeline.py --quick
    
    if [[ $? -eq 0 ]]; then
        print_success "Pipeline completed successfully"
    else
        print_error "Pipeline failed"
        exit 1
    fi
}

# Check if port is available
check_port() {
    local port=$1
    if [[ "$OS_TYPE" == "windows" ]]; then
        # Windows: use netstat
        if netstat -ano | grep ":$port" | grep "LISTENING" >/dev/null 2>&1; then
            return 1
        else
            return 0
        fi
    else
        # Linux/Mac: use lsof
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            return 1
        else
            return 0
        fi
    fi
}

# Kill process on port
kill_port() {
    local port=$1
    print_info "Killing process on port $port..."
    
    if [[ "$OS_TYPE" == "windows" ]]; then
        # Windows: use netstat to find PID and taskkill to kill
        local pid=$(netstat -ano | grep ":$port" | grep "LISTENING" | awk '{print $5}' | head -n 1)
        if [[ -n "$pid" ]]; then
            taskkill //PID $pid //F 2>/dev/null || true
        fi
    else
        # Linux/Mac: use lsof and kill
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
    fi
    
    sleep 1
}

# Start a service in background
start_service() {
    local name=$1
    local command=$2
    local port=$3
    local logfile="logs/${name}.log"
    
    print_info "Starting $name on port $port..."
    
    # Create log directory
    mkdir -p logs
    
    # Start service in background
    nohup $command > $logfile 2>&1 &
    local pid=$!
    
    # Wait for service to start
    if [[ "$OS_TYPE" == "windows" ]]; then
        # Windows: wait longer and check port instead of PID
        sleep 5
        
        # Check if port is listening (more reliable on Windows)
        local attempts=0
        local max_attempts=20
        while [[ $attempts -lt $max_attempts ]]; do
            if netstat -ano | grep ":$port" | grep "LISTENING" >/dev/null 2>&1; then
                print_success "$name started on port $port"
                echo "$name:$port" >> .demo_pids
                return 0
            fi
            attempts=$((attempts + 1))
            sleep 1
        done
        
        print_error "$name failed to start"
        echo "Check logs: tail -f $logfile"
        return 1
    else
        # Linux/Mac: check PID
        sleep 3
        if ps -p $pid > /dev/null 2>&1; then
            print_success "$name started (PID: $pid)"
            echo $pid >> .demo_pids
            return 0
        else
            print_error "$name failed to start"
            echo "Check logs: tail -f $logfile"
            return 1
        fi
    fi
}

# Wait for service to be ready
wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    print_info "Waiting for $name to be ready..."
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s -f "$url" > /dev/null 2>&1 || wget -q --spider "$url" 2>/dev/null; then
            print_success "$name is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 1
    done
    
    echo ""
    print_error "$name failed to become ready"
    return 1
}

# Stop all demo services
stop_demo() {
    print_info "Stopping all demo services..."
    
    if [[ -f ".demo_pids" ]]; then
        while read line; do
            if [[ "$OS_TYPE" == "windows" ]]; then
                # Windows: line format is "name:port", kill by port
                if [[ "$line" == *":"* ]]; then
                    local port=$(echo "$line" | cut -d':' -f2)
                    kill_port $port
                fi
            else
                # Linux/Mac: line is PID
                local pid=$line
                if ps -p $pid > /dev/null 2>/dev/null; then
                    kill $pid 2>/dev/null || true
                    print_info "Stopped process $pid"
                fi
            fi
        done < .demo_pids
        rm .demo_pids
    fi
    
    # Kill common ports as backup
    for port in 8000 8001 8501 8080; do
        if ! check_port $port; then
            kill_port $port
        fi
    done
    
    print_success "All services stopped"
}

# Display access information
show_access_info() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     ACCESS INFORMATION                           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}🎯 Streamlit Multi-page App:${NC}"
    echo -e "   ${BLUE}http://localhost:8501${NC}"
    echo -e "   → Main demo interface with 5 specialized views"
    echo ""
    echo -e "${GREEN}📊 Original Dashboard:${NC}"
    echo -e "   ${BLUE}http://localhost:8080${NC}"
    echo -e "   → Simple HTML dashboard"
    echo ""
    echo -e "${GREEN}🔌 Enhanced API (with WebSocket):${NC}"
    echo -e "   ${BLUE}http://localhost:8001${NC}"
    echo -e "   ${BLUE}http://localhost:8001/docs${NC} (Swagger UI)"
    echo -e "   ${BLUE}ws://localhost:8001/ws${NC} (WebSocket)"
    echo ""
    echo -e "${GREEN}🔌 Original API:${NC}"
    echo -e "   ${BLUE}http://localhost:8000${NC}"
    echo -e "   ${BLUE}http://localhost:8000/docs${NC} (Swagger UI)"
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     QUICK COMMANDS                               ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "View logs:"
    echo "  $ tail -f logs/streamlit.log"
    echo "  $ tail -f logs/api-enhanced.log"
    echo "  $ tail -f logs/api.log"
    echo ""
    echo "Stop all services:"
    echo "  $ ./start_demo.sh --stop"
    echo "  or press Ctrl+C in this terminal"
    echo ""
    echo -e "${GREEN}✓ Decision Intelligence Studio is ready!${NC}"
    echo ""
}

# Main execution
main() {
    # Parse arguments
    if [[ "$1" == "--stop" ]]; then
        print_header
        stop_demo
        exit 0
    fi
    
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        print_header
        echo "Usage: ./start_demo.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --stop    Stop all running demo services"
        echo "  --help    Show this help message"
        echo ""
        echo "Default: Start all demo services"
        exit 0
    fi
    
    # Print header
    print_header
    
    # Pre-flight checks
    print_info "Running pre-flight checks..."
    check_wsl
    check_venv
    
    # Check if data exists
    if ! check_data; then
        print_warning "Running pipeline first (this will take ~2 minutes)..."
        run_pipeline
    fi
    
    # Clean up any existing services
    print_info "Cleaning up existing services..."
    stop_demo
    
    echo ""
    print_info "Starting all services..."
    echo ""
    
    # Initialize PID tracking file
    rm -f .demo_pids
    
    # Start Enhanced API (with WebSocket)
    if ! check_port 8001; then
        kill_port 8001
    fi
    start_service "api-enhanced" "python -m src.api.realtime_api" 8001
    wait_for_service "Enhanced API" "http://localhost:8001/health"
    
    # Start Original API
    if ! check_port 8000; then
        kill_port 8000
    fi
    start_service "api" "python -m src.api.main" 8000
    wait_for_service "API" "http://localhost:8000/health"
    
    # Start Streamlit App
    if ! check_port 8501; then
        kill_port 8501
    fi
    start_service "streamlit" "streamlit run src/streamlit_app/app.py" 8501
    wait_for_service "Streamlit" "http://localhost:8501"
    
    # Start simple HTTP server for dashboard
    if ! check_port 8080; then
        kill_port 8080
    fi
    start_service "dashboard" "python -m http.server 8080 --directory src/ui" 8080
    sleep 2
    
    # Show access information
    show_access_info
    
    # Keep script running and handle Ctrl+C
    trap stop_demo EXIT
    
    print_info "Press Ctrl+C to stop all services"
    
    # Wait forever
    while true; do
        sleep 1
    done
}

# Run main
main "$@"