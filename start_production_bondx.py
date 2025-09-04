#!/usr/bin/env python3
"""
Production BondX Startup Script
Starts both FastAPI backend and React frontend with proper error handling
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting BondX Backend (FastAPI)...")
    try:
        # Change to bondx directory and start backend
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            cwd="bondx",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for startup
        time.sleep(3)
        
        if backend_process.poll() is None:
            print("✅ Backend started successfully on http://localhost:8000")
            print("📚 API Documentation: http://localhost:8000/docs")
            return backend_process
        else:
            stdout, stderr = backend_process.communicate()
            print(f"❌ Backend failed to start:")
            print(f"   STDOUT: {stdout}")
            print(f"   STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the React frontend"""
    print("🎨 Starting BondX Frontend (React + Vite)...")
    try:
        # Start frontend with npm
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for startup
        time.sleep(5)
        
        if frontend_process.poll() is None:
            print("✅ Frontend started successfully")
            return frontend_process
        else:
            stdout, stderr = frontend_process.communicate()
            print(f"❌ Frontend failed to start:")
            print(f"   STDOUT: {stdout}")
            print(f"   STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🚀 BondX Production System Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("bondx").exists():
        print("❌ Error: 'bondx' directory not found")
        print("   Please run this script from the BondX project root")
        sys.exit(1)
    
    if not Path("package.json").exists():
        print("❌ Error: 'package.json' not found")
        print("   Please run this script from the BondX project root")
        sys.exit(1)
    
    # Start backend first
    backend_process = start_backend()
    if not backend_process:
        print("❌ Backend startup failed. Exiting.")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Frontend startup failed. Stopping backend...")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n🎉 BondX Production System Started Successfully!")
    print("=" * 50)
    print("🔗 Frontend Dashboard: http://localhost:3003 (or next available port)")
    print("🔗 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n🔐 Login Credentials:")
    print("   Demo User: demo / demo123")
    print("   Admin User: admin / admin123")
    print("\n💡 Features Available:")
    print("   ✅ Real Corporate Bonds Data (200+ bonds from CSV)")
    print("   ✅ Live Portfolio Analytics")
    print("   ✅ Real-time Risk Metrics")
    print("   ✅ Live Trading Activity")
    print("   ✅ Market Overview with Real Data")
    print("   ✅ Yield Curve Visualization")
    print("   ✅ JWT Authentication")
    print("   ✅ WebSocket Real-time Updates")
    print("\n⏹️  Press Ctrl+C to stop both services")
    
    try:
        # Keep both processes running
        while True:
            time.sleep(1)
            
            # Check if either process has died
            if backend_process.poll() is not None:
                print("❌ Backend process died unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process died unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down BondX...")
        
        # Terminate both processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
            
        print("👋 BondX shutdown complete")

if __name__ == "__main__":
    main()
