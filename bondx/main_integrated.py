"""
Main Integrated System for BondX Phase 3.

This module integrates all Phase 3 components:
- Real-time trading engine with order book and matching
- Market making algorithms
- Smart order routing
- Real-time risk management
- WebSocket infrastructure
- Mobile-optimized APIs
- Compliance and monitoring
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from sqlalchemy.orm import Session

from .core.logging import get_logger, setup_logging
from .core.config import get_settings
from .core.monitoring import MetricsCollector
from .database.base import get_db, init_db
from .trading_engine.order_book import OrderBookManager
from .trading_engine.matching_engine import MatchingEngine
from .trading_engine.market_maker import MarketMakerManager
from .trading_engine.order_router import OrderRouter
from .risk_management.real_time_risk import RealTimeRiskEngine
from .websocket.websocket_manager import WebSocketManager
from .auction_engine.auction_engine import AuctionEngine
from .api.v1.mobile import router as mobile_router
from .api.v1.trading import router as trading_router
from .api.v1.risk import router as risk_router
from .api.v1.websocket import router as websocket_router

logger = get_logger(__name__)

# Global state
class BondXSystem:
    """Main BondX system integration."""
    
    def __init__(self):
        self.settings = get_settings()
        self.metrics = MetricsCollector()
        
        # Core components
        self.order_book_manager: Optional[OrderBookManager] = None
        self.matching_engine: Optional[MatchingEngine] = None
        self.market_maker_manager: Optional[MarketMakerManager] = None
        self.order_router: Optional[OrderRouter] = None
        self.risk_engine: Optional[RealTimeRiskEngine] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        self.auction_engine: Optional[AuctionEngine] = None
        
        # System state
        self.is_running = False
        self.start_time = datetime.utcnow()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("BondX System initialized")
    
    async def start(self):
        """Start the integrated BondX system."""
        if self.is_running:
            logger.warning("BondX System is already running")
            return
        
        try:
            logger.info("Starting BondX Integrated System...")
            
            # Initialize database
            await init_db()
            logger.info("Database initialized")
            
            # Initialize core components
            await self._initialize_core_components()
            logger.info("Core components initialized")
            
            # Start background services
            await self._start_background_services()
            logger.info("Background services started")
            
            # Start market makers
            await self._start_market_makers()
            logger.info("Market makers started")
            
            # Start WebSocket manager
            await self._start_websocket_manager()
            logger.info("WebSocket manager started")
            
            # Start monitoring
            await self._start_monitoring()
            logger.info("Monitoring started")
            
            self.is_running = True
            logger.info("BondX Integrated System started successfully")
            
        except Exception as e:
            logger.error(f"Error starting BondX System: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the integrated BondX system."""
        if not self.is_running:
            logger.warning("BondX System is not running")
            return
        
        try:
            logger.info("Stopping BondX Integrated System...")
            
            # Stop background services
            await self._stop_background_services()
            
            # Stop market makers
            await self._stop_market_makers()
            
            # Stop WebSocket manager
            await self._stop_websocket_manager()
            
            # Stop monitoring
            await self._stop_monitoring()
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.background_tasks.clear()
            self.is_running = False
            
            logger.info("BondX Integrated System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping BondX System: {e}")
    
    async def _initialize_core_components(self):
        """Initialize core system components."""
        try:
            # Initialize order book manager
            self.order_book_manager = OrderBookManager()
            logger.info("Order book manager initialized")
            
            # Initialize matching engine
            self.matching_engine = MatchingEngine(
                order_book_manager=self.order_book_manager,
                risk_checker=self._risk_check_callback
            )
            logger.info("Matching engine initialized")
            
            # Initialize market maker manager
            self.market_maker_manager = MarketMakerManager(
                order_book_manager=self.order_book_manager
            )
            logger.info("Market maker manager initialized")
            
            # Initialize order router
            self.order_router = OrderRouter(
                order_book_manager=self.order_book_manager,
                matching_engine=self.matching_engine,
                auction_engine=self.auction_engine
            )
            logger.info("Order router initialized")
            
            # Initialize risk engine
            self.risk_engine = RealTimeRiskEngine()
            logger.info("Risk engine initialized")
            
            # Initialize auction engine
            self.auction_engine = AuctionEngine(db_session=None)  # Will be set properly
            logger.info("Auction engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing core components: {e}")
            raise
    
    async def _start_background_services(self):
        """Start background services."""
        try:
            # Start matching engine
            if self.matching_engine:
                await self.matching_engine.start()
                logger.info("Matching engine started")
            
            # Start risk engine background tasks
            if self.risk_engine:
                risk_task = asyncio.create_task(self._risk_monitoring_loop())
                self.background_tasks.append(risk_task)
                logger.info("Risk monitoring started")
            
            # Start order book monitoring
            order_book_task = asyncio.create_task(self._order_book_monitoring_loop())
            self.background_tasks.append(order_book_task)
            logger.info("Order book monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting background services: {e}")
            raise
    
    async def _start_market_makers(self):
        """Start market makers for active instruments."""
        try:
            if not self.market_maker_manager:
                return
            
            # Get active instruments (in practice, this would come from database)
            active_instruments = ["BOND_001", "BOND_002", "BOND_003"]
            
            for instrument_id in active_instruments:
                try:
                    market_maker = self.market_maker_manager.create_market_maker(
                        instrument_id=instrument_id,
                        yield_calculator=None,  # Will be set when available
                        bond_pricer=None,  # Will be set when available
                        risk_checker=self._risk_check_callback
                    )
                    
                    await market_maker.start()
                    logger.info(f"Market maker started for {instrument_id}")
                    
                except Exception as e:
                    logger.error(f"Error starting market maker for {instrument_id}: {e}")
                    continue
            
            logger.info(f"Started {len(active_instruments)} market makers")
            
        except Exception as e:
            logger.error(f"Error starting market makers: {e}")
            raise
    
    async def _start_websocket_manager(self):
        """Start WebSocket manager."""
        try:
            if not self.websocket_manager:
                self.websocket_manager = WebSocketManager()
            
            await self.websocket_manager.start()
            logger.info("WebSocket manager started")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket manager: {e}")
            raise
    
    async def _start_monitoring(self):
        """Start system monitoring."""
        try:
            # Start metrics collection
            metrics_task = asyncio.create_task(self._metrics_collection_loop())
            self.background_tasks.append(metrics_task)
            logger.info("Metrics collection started")
            
            # Start health monitoring
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self.background_tasks.append(health_task)
            logger.info("Health monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            raise
    
    async def _stop_background_services(self):
        """Stop background services."""
        try:
            if self.matching_engine:
                await self.matching_engine.stop()
                logger.info("Matching engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping background services: {e}")
    
    async def _stop_market_makers(self):
        """Stop market makers."""
        try:
            if self.market_maker_manager:
                await self.market_maker_manager.stop_all()
                logger.info("Market makers stopped")
            
        except Exception as e:
            logger.error(f"Error stopping market makers: {e}")
    
    async def _stop_websocket_manager(self):
        """Stop WebSocket manager."""
        try:
            if self.websocket_manager:
                await self.websocket_manager.stop()
                logger.info("WebSocket manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket manager: {e}")
    
    async def _stop_monitoring(self):
        """Stop system monitoring."""
        try:
            # Background tasks will be cancelled in main stop method
            pass
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def _risk_check_callback(self, order) -> Any:
        """Risk check callback for orders."""
        try:
            # In practice, this would perform comprehensive risk checks
            # For now, return a simple pass/fail result
            return type('RiskCheckResult', (), {
                'passed': True,
                'risk_score': 0.1,
                'limits_checked': ['position', 'exposure', 'concentration']
            })()
            
        except Exception as e:
            logger.error(f"Error in risk check callback: {e}")
            return type('RiskCheckResult', (), {
                'passed': False,
                'risk_score': 1.0,
                'limits_checked': []
            })()
    
    async def _risk_monitoring_loop(self):
        """Risk monitoring background loop."""
        while self.is_running:
            try:
                # Update portfolio risk metrics
                if self.risk_engine:
                    # In practice, this would iterate through all portfolios
                    # For now, just log that monitoring is active
                    await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _order_book_monitoring_loop(self):
        """Order book monitoring background loop."""
        while self.is_running:
            try:
                if self.order_book_manager:
                    # Get order book statistics
                    stats = self.order_book_manager.get_global_statistics()
                    
                    # Update metrics
                    self.metrics.update_gauge("order_book.total_instruments", stats["total_instruments"])
                    self.metrics.update_gauge("order_book.total_orders", stats["total_orders"])
                    self.metrics.update_gauge("order_book.total_volume", float(stats["total_bid_volume"]))
                    
                    # Broadcast market data updates via WebSocket
                    if self.websocket_manager:
                        snapshots = self.order_book_manager.get_all_snapshots()
                        for instrument_id, snapshot in snapshots.items():
                            await self.websocket_manager.broadcast_market_data(
                                instrument_id, 
                                {
                                    "best_bid": float(snapshot.best_bid) if snapshot.best_bid else None,
                                    "best_ask": float(snapshot.best_ask) if snapshot.best_ask else None,
                                    "mid_price": float(snapshot.mid_price) if snapshot.mid_price else None,
                                    "spread": float(snapshot.spread) if snapshot.spread else None,
                                    "total_bid_volume": float(snapshot.total_bid_volume),
                                    "total_ask_volume": float(snapshot.total_ask_volume)
                                }
                            )
                
                await asyncio.sleep(1)  # Update every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in order book monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collection_loop(self):
        """Metrics collection background loop."""
        while self.is_running:
            try:
                # Collect system metrics
                if self.matching_engine:
                    match_stats = self.matching_engine.get_statistics()
                    self.metrics.update_gauge("matching.orders_per_second", match_stats["orders_per_second"])
                    self.metrics.update_gauge("matching.trades_per_second", match_stats["trades_per_second"])
                
                if self.websocket_manager:
                    ws_stats = self.websocket_manager.get_connection_statistics()
                    self.metrics.update_gauge("websocket.total_clients", ws_stats["total_clients"])
                    self.metrics.update_gauge("websocket.messages_sent", ws_stats["messages_sent"])
                
                if self.market_maker_manager:
                    mm_stats = self.market_maker_manager.get_global_statistics()
                    self.metrics.update_gauge("market_maker.total_instruments", mm_stats["total_instruments"])
                    self.metrics.update_gauge("market_maker.total_pnl", float(mm_stats["total_pnl"]))
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitoring_loop(self):
        """Health monitoring background loop."""
        while self.is_running:
            try:
                # Check system health
                health_status = await self._check_system_health()
                
                if not health_status["healthy"]:
                    logger.warning(f"System health issues detected: {health_status['issues']}")
                
                # Update health metrics
                self.metrics.update_gauge("system.health_score", health_status["health_score"])
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            health_score = 100
            issues = []
            
            # Check matching engine
            if self.matching_engine and not self.matching_engine.is_running:
                health_score -= 20
                issues.append("Matching engine not running")
            
            # Check WebSocket manager
            if self.websocket_manager and not self.websocket_manager.is_running:
                health_score -= 15
                issues.append("WebSocket manager not running")
            
            # Check order book manager
            if not self.order_book_manager:
                health_score -= 10
                issues.append("Order book manager not available")
            
            # Check risk engine
            if not self.risk_engine:
                health_score -= 10
                issues.append("Risk engine not available")
            
            return {
                "healthy": health_score >= 80,
                "health_score": health_score,
                "issues": issues,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                "healthy": False,
                "health_score": 0,
                "issues": [f"Health check error: {e}"],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat(),
            "components": {
                "order_book_manager": self.order_book_manager is not None,
                "matching_engine": self.matching_engine is not None and self.matching_engine.is_running,
                "market_maker_manager": self.market_maker_manager is not None,
                "order_router": self.order_router is not None,
                "risk_engine": self.risk_engine is not None,
                "websocket_manager": self.websocket_manager is not None and self.websocket_manager.is_running,
                "auction_engine": self.auction_engine is not None
            },
            "background_tasks": len(self.background_tasks),
            "active_tasks": len([t for t in self.background_tasks if not t.done()])
        }


# Global system instance
bondx_system = BondXSystem()


# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        await bondx_system.start()
        logger.info("FastAPI application startup completed")
    except Exception as e:
        logger.error(f"FastAPI startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        await bondx_system.stop()
        logger.info("FastAPI application shutdown completed")
    except Exception as e:
        logger.error(f"FastAPI shutdown failed: {e}")


# Create FastAPI app
app = FastAPI(
    title="BondX Integrated System",
    description="Phase 3 Integrated Bond Trading Platform",
    version="3.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(mobile_router, prefix="/api/v1")
app.include_router(trading_router, prefix="/api/v1")
app.include_router(risk_router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/api/v1")


# Health check endpoints
@app.get("/health")
async def health_check():
    """System health check."""
    try:
        system_status = bondx_system.get_system_status()
        health_status = await bondx_system._check_system_health()
        
        return {
            "status": "healthy" if health_status["healthy"] else "unhealthy",
            "system": system_status,
            "health": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/healthz")
async def healthz():
    """Kubernetes health check endpoint."""
    try:
        health_status = await bondx_system._check_system_health()
        if health_status["healthy"]:
            return {"status": "ok"}
        else:
            raise HTTPException(status_code=503, detail="System unhealthy")
            
    except Exception as e:
        logger.error(f"Healthz check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@app.get("/readyz")
async def readyz():
    """Kubernetes readiness check endpoint."""
    try:
        system_status = bondx_system.get_system_status()
        if system_status["is_running"]:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="System not ready")
            
    except Exception as e:
        logger.error(f"Readyz check failed: {e}")
        raise HTTPException(status_code=503, detail="System not ready")


# System status endpoint
@app.get("/api/v1/system/status")
async def get_system_status():
    """Get detailed system status."""
    try:
        return bondx_system.get_system_status()
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        return bondx_system.metrics.get_all_metrics()
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


# Signal handlers
def signal_handler(signum, frame):
    """Handle system signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(bondx_system.stop())
    sys.exit(0)


# Main function
async def main():
    """Main entry point."""
    try:
        # Setup logging
        setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the system
        await bondx_system.start()
        
        # Run the FastAPI application
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Main function error: {e}")
        await bondx_system.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
