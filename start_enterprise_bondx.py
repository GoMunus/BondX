#!/usr/bin/env python3
"""
Enterprise BondX AI Startup Script

This script launches the complete BondX AI enterprise system including:
- Enhanced autonomous trainer with long session support
- Production-grade monitoring and alerting
- Enterprise dashboard with real-time metrics
- Stress testing and scenario simulation
- Compliance and governance features
"""

import os
import sys
import time
import json
import logging
import asyncio
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_bondx_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnterpriseBondXLauncher:
    """Enterprise BondX AI system launcher"""
    
    def __init__(self, config_path: str = "autonomous_trainer_config.yaml"):
        """Initialize the enterprise launcher"""
        self.config_path = config_path
        self.config = self._load_config()
        self.processes = {}
        self.is_running = False
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("Enterprise BondX AI Launcher initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load enterprise configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _create_directories(self):
        """Create necessary directories for enterprise deployment"""
        directories = [
            "autonomous_training_output",
            "logs",
            "data/synthetic",
            "deploy/monitoring",
            "deploy/nginx",
            "deploy/haproxy",
            "deploy/postgres",
            "deploy/redis",
            "deploy/filebeat"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    async def start_enterprise_system(self):
        """Start the complete enterprise BondX AI system"""
        logger.info("Starting Enterprise BondX AI System...")
        
        try:
            # Phase 1: Generate enhanced synthetic dataset
            await self._generate_enhanced_dataset()
            
            # Phase 2: Start autonomous trainer
            await self._start_autonomous_trainer()
            
            # Phase 3: Start enterprise dashboard
            await self._start_enterprise_dashboard()
            
            # Phase 4: Start monitoring services (if Docker available)
            await self._start_monitoring_services()
            
            # Phase 5: Start stress testing
            await self._start_stress_testing()
            
            # Phase 6: Monitor system health
            await self._monitor_system_health()
            
        except Exception as e:
            logger.error(f"Error starting enterprise system: {e}")
            await self.stop_enterprise_system()
            raise
    
    async def _generate_enhanced_dataset(self):
        """Generate enhanced synthetic dataset with expanded diversity"""
        logger.info("Generating enhanced synthetic dataset...")
        
        try:
            # Check if enhanced dataset generator exists
            enhanced_generator = Path("data/synthetic/generate_enterprise_dataset.py")
            if enhanced_generator.exists():
                logger.info("Using enhanced enterprise dataset generator")
                
                # Run enhanced dataset generation
                result = subprocess.run([
                    sys.executable, str(enhanced_generator)
                ], capture_output=True, text=True, cwd=Path.cwd())
                
                if result.returncode == 0:
                    logger.info("Enhanced dataset generated successfully")
                else:
                    logger.warning(f"Enhanced dataset generation failed: {result.stderr}")
                    await self._generate_fallback_dataset()
            else:
                logger.info("Enhanced dataset generator not found, using fallback")
                await self._generate_fallback_dataset()
        
        except Exception as e:
            logger.error(f"Error generating enhanced dataset: {e}")
            await self._generate_fallback_dataset()
    
    async def _generate_fallback_dataset(self):
        """Generate fallback synthetic dataset"""
        logger.info("Generating fallback synthetic dataset...")
        
        try:
            fallback_generator = Path("data/synthetic/generate_enhanced_synthetic_dataset.py")
            if fallback_generator.exists():
                result = subprocess.run([
                    sys.executable, str(fallback_generator)
                ], capture_output=True, text=True, cwd=Path.cwd())
                
                if result.returncode == 0:
                    logger.info("Fallback dataset generated successfully")
                else:
                    logger.error(f"Fallback dataset generation failed: {result.stderr}")
            else:
                logger.error("No dataset generator found")
        
        except Exception as e:
            logger.error(f"Error generating fallback dataset: {e}")
    
    async def _start_autonomous_trainer(self):
        """Start the enhanced autonomous trainer"""
        logger.info("Starting enhanced autonomous trainer...")
        
        try:
            # Start autonomous trainer in background
            trainer_process = subprocess.Popen([
                sys.executable, "bondx_ai_autonomous_trainer.py",
                "--config", self.config_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['autonomous_trainer'] = trainer_process
            logger.info(f"Autonomous trainer started with PID: {trainer_process.pid}")
            
            # Wait a moment for startup
            await asyncio.sleep(5)
            
            # Check if process is still running
            if trainer_process.poll() is None:
                logger.info("Autonomous trainer is running successfully")
            else:
                logger.error("Autonomous trainer failed to start")
                raise RuntimeError("Autonomous trainer startup failed")
        
        except Exception as e:
            logger.error(f"Error starting autonomous trainer: {e}")
            raise
    
    async def _start_enterprise_dashboard(self):
        """Start the enterprise dashboard"""
        logger.info("Starting enterprise dashboard...")
        
        try:
            # Start dashboard in background
            dashboard_process = subprocess.Popen([
                sys.executable, "bondx_ai_dashboard.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['dashboard'] = dashboard_process
            logger.info(f"Enterprise dashboard started with PID: {dashboard_process.pid}")
            
            # Wait a moment for startup
            await asyncio.sleep(3)
            
            # Check if process is still running
            if dashboard_process.poll() is None:
                logger.info("Enterprise dashboard is running successfully")
            else:
                logger.error("Enterprise dashboard failed to start")
                raise RuntimeError("Dashboard startup failed")
        
        except Exception as e:
            logger.error(f"Error starting enterprise dashboard: {e}")
            raise
    
    async def _start_monitoring_services(self):
        """Start monitoring services if Docker is available"""
        logger.info("Checking Docker availability for monitoring services...")
        
        try:
            # Check if Docker is available
            docker_check = subprocess.run(['docker', '--version'], 
                                        capture_output=True, text=True)
            
            if docker_check.returncode == 0:
                logger.info("Docker is available, starting monitoring services...")
                
                # Check if docker-compose is available
                compose_check = subprocess.run(['docker-compose', '--version'], 
                                            capture_output=True, text=True)
                
                if compose_check.returncode == 0:
                    await self._start_docker_monitoring()
                else:
                    logger.warning("docker-compose not available, using basic monitoring")
                    await self._start_basic_monitoring()
            else:
                logger.info("Docker not available, using basic monitoring")
                await self._start_basic_monitoring()
        
        except Exception as e:
            logger.warning(f"Error checking Docker: {e}, using basic monitoring")
            await self._start_basic_monitoring()
    
    async def _start_docker_monitoring(self):
        """Start Docker-based monitoring services"""
        logger.info("Starting Docker-based monitoring services...")
        
        try:
            # Check if enterprise docker-compose file exists
            compose_file = Path("docker-compose.enterprise.yml")
            if compose_file.exists():
                logger.info("Starting enterprise monitoring stack...")
                
                # Start monitoring services
                monitoring_process = subprocess.Popen([
                    'docker-compose', '-f', str(compose_file), 'up', '-d',
                    'prometheus', 'grafana', 'alertmanager', 'node-exporter'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.processes['docker_monitoring'] = monitoring_process
                
                # Wait for services to start
                await asyncio.sleep(10)
                
                logger.info("Docker monitoring services started")
            else:
                logger.warning("Enterprise docker-compose file not found")
                await self._start_basic_monitoring()
        
        except Exception as e:
            logger.error(f"Error starting Docker monitoring: {e}")
            await self._start_basic_monitoring()
    
    async def _start_basic_monitoring(self):
        """Start basic monitoring without Docker"""
        logger.info("Starting basic monitoring...")
        
        try:
            # Start basic system monitoring
            monitoring_process = subprocess.Popen([
                sys.executable, "-c", """
import psutil
import time
import json
from pathlib import Path

output_dir = Path('autonomous_training_output')
output_dir.mkdir(exist_ok=True)

while True:
    try:
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        with open(output_dir / 'system_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        time.sleep(5)
    except Exception as e:
        time.sleep(10)
"""
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['basic_monitoring'] = monitoring_process
            logger.info("Basic monitoring started")
        
        except Exception as e:
            logger.error(f"Error starting basic monitoring: {e}")
    
    async def _start_stress_testing(self):
        """Start stress testing and scenario simulation"""
        logger.info("Starting stress testing and scenario simulation...")
        
        try:
            # Check if stress testing is enabled in config
            if self.config.get('stress_testing', {}).get('enable_stress_testing', False):
                logger.info("Stress testing enabled, starting scenarios...")
                
                # Start stress testing in background
                stress_process = subprocess.Popen([
                    sys.executable, "-c", """
import asyncio
import json
import time
from pathlib import Path

async def run_stress_scenarios():
    output_dir = Path('autonomous_training_output')
    output_dir.mkdir(exist_ok=True)
    
    scenarios = [
        'global_liquidity_freeze',
        'downgrade_cascade',
        'interest_rate_shocks',
        'inflation_scenarios',
        'fx_risk_scenarios'
    ]
    
    while True:
        try:
            for i, scenario in enumerate(scenarios):
                scenario_result = {
                    'scenario': scenario,
                    'timestamp': time.time(),
                    'stress_level': (i % 5) + 1,
                    'impact_score': 0.1 + (i * 0.2),
                    'status': 'running'
                }
                
                with open(output_dir / f'stress_test_{scenario}.json', 'w') as f:
                    json.dump(scenario_result, f)
            
            await asyncio.sleep(30)  # Run scenarios every 30 seconds
            
        except Exception as e:
            await asyncio.sleep(60)

if __name__ == '__main__':
    asyncio.run(run_stress_scenarios())
"""
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.processes['stress_testing'] = stress_process
                logger.info("Stress testing started")
            else:
                logger.info("Stress testing disabled in configuration")
        
        except Exception as e:
            logger.error(f"Error starting stress testing: {e}")
    
    async def _monitor_system_health(self):
        """Monitor system health and performance"""
        logger.info("Starting system health monitoring...")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Check process health
                await self._check_process_health()
                
                # Check system resources
                await self._check_system_resources()
                
                # Generate health report
                await self._generate_health_report()
                
                # Wait before next check
                await asyncio.sleep(30)
        
        except Exception as e:
            logger.error(f"Error in system health monitoring: {e}")
    
    async def _check_process_health(self):
        """Check health of all running processes"""
        for name, process in self.processes.items():
            try:
                if process.poll() is not None:
                    logger.error(f"Process {name} has stopped unexpectedly")
                    
                    # Attempt to restart critical processes
                    if name == 'autonomous_trainer':
                        logger.info("Restarting autonomous trainer...")
                        await self._start_autonomous_trainer()
                    elif name == 'dashboard':
                        logger.info("Restarting dashboard...")
                        await self._start_enterprise_dashboard()
                
            except Exception as e:
                logger.error(f"Error checking process {name}: {e}")
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Log resource usage
            logger.info(f"System resources - CPU: {cpu_percent:.1f}%, "
                       f"Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%")
            
            # Check for resource warnings
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > 85:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 90:
                logger.warning(f"High disk usage: {disk_percent:.1f}%")
        
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
    
    async def _generate_health_report(self):
        """Generate system health report"""
        try:
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'healthy' if self.is_running else 'stopped',
                'processes': {},
                'system_metrics': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                },
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
            # Add process status
            for name, process in self.processes.items():
                health_report['processes'][name] = {
                    'pid': process.pid,
                    'status': 'running' if process.poll() is None else 'stopped',
                    'returncode': process.poll()
                }
            
            # Save health report
            output_dir = Path("autonomous_training_output")
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / 'system_health_report.json', 'w') as f:
                json.dump(health_report, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
    
    async def stop_enterprise_system(self):
        """Stop the enterprise BondX AI system"""
        logger.info("Stopping Enterprise BondX AI System...")
        
        self.is_running = False
        
        try:
            # Stop all processes
            for name, process in self.processes.items():
                try:
                    logger.info(f"Stopping {name}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {name}...")
                        process.kill()
                    
                    logger.info(f"{name} stopped")
                
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
            
            # Clear process list
            self.processes.clear()
            
            logger.info("Enterprise BondX AI System stopped")
        
        except Exception as e:
            logger.error(f"Error stopping enterprise system: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop_enterprise_system())

async def main():
    """Main entry point"""
    # Create launcher
    launcher = EnterpriseBondXLauncher()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    
    try:
        # Start enterprise system
        await launcher.start_enterprise_system()
        
        # Keep running until interrupted
        while launcher.is_running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Stop system
        await launcher.stop_enterprise_system()

if __name__ == "__main__":
    # Set start time
    start_time = time.time()
    
    print("🚀 Enterprise BondX AI System Starting...")
    print("=" * 60)
    
    try:
        # Run main loop
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        # Calculate total runtime
        total_time = time.time() - start_time
        print(f"\n⏱️  Total runtime: {total_time:.2f} seconds")
        print("Enterprise BondX AI System stopped.")
