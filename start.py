#!/usr/bin/env python3
"""
Startup script for BondX Backend.

This script provides a simple way to start the application
with proper configuration and error handling.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    """Main startup function."""
    try:
        # Import the application
        from bondx.main import app
        
        # Import uvicorn
        import uvicorn
        
        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "false").lower() == "true"
        log_level = os.getenv("LOG_LEVEL", "info")
        
        print(f"🚀 Starting BondX Backend...")
        print(f"📍 Host: {host}")
        print(f"🔌 Port: {port}")
        print(f"🔄 Reload: {reload}")
        print(f"📝 Log Level: {log_level}")
        print(f"🌐 API Documentation: http://{host}:{port}/docs")
        print(f"❤️  Health Check: http://{host}:{port}/health")
        
        # Start the server
        uvicorn.run(
            "bondx.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        print("   or")
        print("   poetry install")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
