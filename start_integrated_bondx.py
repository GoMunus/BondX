#!/usr/bin/env python3
"""
BondX Integrated System Startup Script

This script starts the complete BondX system including:
1. FastAPI backend with all engines
2. Frontend development server (if in development mode)
3. WebSocket managers
4. Background tasks for real-time data

Usage:
    python start_integrated_bondx.py [--dev] [--frontend-only] [--backend-only]
"""

import asyncio
import subprocess
import sys
import os
import signal
import time
from pathlib import Path
from typing import List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_command(command: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> subprocess.Popen:
    """Run a command in a subprocess."""
    print(f"Starting: {' '.join(command)}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import redis
        print("✅ Python dependencies found")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check Node.js dependencies (if in dev mode)
    if os.path.exists(project_root / "node_modules"):
        print("✅ Node.js dependencies found")
    else:
        print("⚠️  Node.js dependencies not found")
        print("Run: npm install")
    
    return True

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting BondX Backend...")
    
    backend_env = os.environ.copy()
    backend_env.update({
        'PYTHONPATH': str(project_root),
        'BONDX_ENV': 'development',
        'BONDX_DEBUG': 'true'
    })
    
    # Start FastAPI with uvicorn
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "bondx.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ]
    
    return run_command(backend_cmd, cwd=project_root, env=backend_env)

def start_frontend():
    """Start the frontend development server."""
    print("🎨 Starting BondX Frontend...")
    
    # Check if npm is available
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm not found. Please install Node.js")
        return None
    
    # Start Vite development server
    frontend_cmd = ["npm", "run", "dev"]
    
    return run_command(frontend_cmd, cwd=project_root)

def monitor_processes(processes: List[subprocess.Popen]):
    """Monitor running processes and handle output."""
    print("📊 Monitoring processes...")
    
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down BondX...")
        for proc in processes:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Force kill if still running
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
        
        print("✅ BondX shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    while True:
        # Check if any process has died
        for i, proc in enumerate(processes):
            if proc.poll() is not None:
                print(f"❌ Process {i} has stopped")
                return
        
        # Read and display output
        for i, proc in enumerate(processes):
            if proc.stdout and proc.stdout.readable():
                try:
                    line = proc.stdout.readline()
                    if line:
                        service_name = "Backend" if i == 0 else "Frontend"
                        print(f"[{service_name}] {line.strip()}")
                except:
                    pass
        
        time.sleep(0.1)

def main():
    """Main function to start the integrated BondX system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start BondX Integrated System")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    parser.add_argument("--frontend-only", action="store_true", help="Start frontend only")
    parser.add_argument("--backend-only", action="store_true", help="Start backend only")
    parser.add_argument("--port", type=int, default=8000, help="Backend port")
    parser.add_argument("--host", default="0.0.0.0", help="Backend host")
    
    args = parser.parse_args()
    
    print("🏦 BondX - Institutional Bond Trading Platform")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    processes = []
    
    try:
        # Start backend
        if not args.frontend_only:
            backend_proc = start_backend()
            if backend_proc:
                processes.append(backend_proc)
                print("✅ Backend started successfully")
                time.sleep(3)  # Wait for backend to start
        
        # Start frontend
        if not args.backend_only and args.dev:
            frontend_proc = start_frontend()
            if frontend_proc:
                processes.append(frontend_proc)
                print("✅ Frontend started successfully")
        
        if not processes:
            print("❌ No processes started")
            return
        
        # Display startup information
        print("\n" + "=" * 50)
        print("🎉 BondX System is running!")
        print("=" * 50)
        
        if not args.frontend_only:
            print("📡 Backend API: http://localhost:8000")
            print("📚 API Docs: http://localhost:8000/docs")
            print("🔌 WebSocket: ws://localhost:8000/ws")
        
        if not args.backend_only and args.dev:
            print("🎨 Frontend: http://localhost:5173")
        
        print("\n🔐 Demo Login Credentials:")
        print("  Username: demo, Password: demo123")
        print("  Username: admin, Password: admin123")
        print("  Username: portfolio_manager, Password: pm123")
        
        print("\n📊 Available Endpoints:")
        print("  - Portfolio Summary: GET /api/v1/dashboard/portfolio/summary")
        print("  - Risk Metrics: GET /api/v1/dashboard/risk/metrics")
        print("  - Trading Activity: GET /api/v1/dashboard/trading/activity")
        print("  - Market Status: GET /api/v1/dashboard/market/status")
        print("  - Yield Curve: GET /api/v1/dashboard/yield-curve")
        
        print("\n🌐 WebSocket Endpoints:")
        print("  - Dashboard: ws://localhost:8000/api/v1/ws/dashboard/connect")
        print("  - Trades: ws://localhost:8000/api/v1/ws/dashboard/trades")
        print("  - Risk Alerts: ws://localhost:8000/api/v1/ws/dashboard/risk-alerts")
        
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)
        
        # Monitor processes
        monitor_processes(processes)
        
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
