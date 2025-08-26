#!/usr/bin/env python3
"""
Simple test script to verify WebSocket system functionality.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_websocket_imports():
    """Test that all WebSocket components can be imported."""
    try:
        print("Testing WebSocket system imports...")
        
        # Test WebSocket router import
        from bondx.api.v1.websocket import router
        print("✓ WebSocket router imported successfully")
        
        # Test unified manager import
        from bondx.websocket.unified_websocket_manager import (
            UnifiedWebSocketManager,
            WebSocketConnection,
            WebSocketMessage,
            MessageType
        )
        print("✓ Unified WebSocket manager imported successfully")
        
        # Test monitoring import
        from websocket_monitoring import WebSocketMonitor, check_websocket_health
        print("✓ WebSocket monitoring imported successfully")
        
        print("\nAll WebSocket components imported successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

async def test_websocket_manager_creation():
    """Test that WebSocket manager can be created."""
    try:
        print("\nTesting WebSocket manager creation...")
        
        from bondx.websocket.unified_websocket_manager import UnifiedWebSocketManager
        
        # Create manager instance
        manager = UnifiedWebSocketManager()
        print("✓ WebSocket manager created successfully")
        
        # Test basic methods
        stats = await manager.get_statistics()
        print(f"✓ Manager statistics: {stats}")
        
        topics = await manager.list_topics()
        print(f"✓ Manager topics: {topics}")
        
        print("✓ WebSocket manager functionality verified!")
        return True
        
    except Exception as e:
        print(f"✗ WebSocket manager error: {e}")
        return False

async def test_router_endpoints():
    """Test that router has the expected endpoints."""
    try:
        print("\nTesting WebSocket router endpoints...")
        
        from bondx.api.v1.websocket import router
        
        # Check that router exists
        if router is not None:
            print("✓ WebSocket router exists")
        else:
            print("✗ WebSocket router is None")
            return False
        
        # Check router prefix
        if router.prefix == "/ws":
            print("✓ Router prefix is correct: /ws")
        else:
            print(f"✗ Router prefix is incorrect: {router.prefix}")
            return False
        
        # Check router tags
        if "websocket" in router.tags:
            print("✓ Router tags are correct")
        else:
            print(f"✗ Router tags are incorrect: {router.tags}")
            return False
        
        print("✓ WebSocket router endpoints verified!")
        return True
        
    except Exception as e:
        print(f"✗ Router endpoint error: {e}")
        return False

async def main():
    """Main test function."""
    print("=" * 50)
    print("BondX WebSocket System Test")
    print("=" * 50)
    
    tests = [
        test_websocket_imports(),
        test_websocket_manager_creation(),
        test_router_endpoints()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Test {i+1}: ✗ Failed with exception: {result}")
        elif result:
            print(f"Test {i+1}: ✓ Passed")
            passed += 1
        else:
            print(f"Test {i+1}: ✗ Failed")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! WebSocket system is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        sys.exit(1)
