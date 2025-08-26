#!/usr/bin/env python3
"""
Startup script to verify WebSocket system initialization.
"""

import asyncio
import sys
import os
import signal

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def start_websocket_system():
    """Start the WebSocket system and verify it's working."""
    try:
        print("Starting BondX WebSocket system...")
        
        # Import components
        from bondx.websocket.unified_websocket_manager import UnifiedWebSocketManager
        from bondx.api.v1.websocket import router
        
        print("✓ Components imported successfully")
        
        # Create WebSocket manager
        manager = UnifiedWebSocketManager()
        print("✓ WebSocket manager created")
        
        # Start the manager
        await manager.start()
        print("✓ WebSocket manager started")
        
        # Get initial statistics
        stats = await manager.get_statistics()
        print(f"✓ Initial statistics: {stats}")
        
        # List topics
        topics = await manager.list_topics()
        print(f"✓ Available topics: {topics}")
        
        print("\n🎉 WebSocket system started successfully!")
        print("System is ready to accept connections.")
        
        return manager
        
    except Exception as e:
        print(f"❌ Failed to start WebSocket system: {e}")
        raise

async def run_system(manager):
    """Run the system for a short time to verify functionality."""
    try:
        print("\nRunning system for 10 seconds to verify functionality...")
        
        # Run for 10 seconds
        for i in range(10):
            await asyncio.sleep(1)
            stats = await manager.get_statistics()
            print(f"  {10-i}s remaining - Active connections: {stats['connections']['active']}")
        
        print("✓ System ran successfully for 10 seconds")
        
    except Exception as e:
        print(f"❌ Error during system run: {e}")

async def shutdown_system(manager):
    """Shutdown the WebSocket system gracefully."""
    try:
        print("\nShutting down WebSocket system...")
        await manager.stop()
        print("✓ WebSocket system shut down successfully")
        
    except Exception as e:
        print(f"❌ Error during shutdown: {e}")

async def main():
    """Main function."""
    manager = None
    
    try:
        # Start the system
        manager = await start_websocket_system()
        
        # Run the system
        await run_system(manager)
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Always try to shutdown
        if manager:
            await shutdown_system(manager)
        
        print("\nWebSocket system test completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
