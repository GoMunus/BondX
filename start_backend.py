#!/usr/bin/env python3
"""
BondX Backend Server Starter

This script starts the BondX backend server with proper configuration
for frontend integration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from bondx.main_integrated import app

def main():
    """Start the BondX backend server."""
    print("🚀 Starting BondX Backend Server...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔗 API Base URL: http://localhost:8000/api/v1")
    print("💚 Health Check: http://localhost:8000/health")
    print("📡 WebSocket: ws://localhost:8000/api/v1/ws")
    print("\n" + "="*50)
    
    # Start the server
    uvicorn.run(
        "bondx.main_integrated:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
