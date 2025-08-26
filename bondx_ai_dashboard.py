#!/usr/bin/env python3
"""
BondX AI Real-Time Training Dashboard - Enhanced for Enterprise

This dashboard provides real-time monitoring of the autonomous training system,
displaying training progress, quality metrics, and convergence status.
Enhanced with production-grade monitoring, Prometheus metrics, and enterprise alerting.
"""

import os
import sys
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import psutil
import threading
from dataclasses import dataclass

# Add bondx to path
sys.path.append(str(Path(__file__).parent / "bondx"))

# Configure enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    timestamp: datetime

@dataclass
class Alert:
    """Alert information"""
    level: str  # 'info', 'warning', 'error', 'critical'
    message: str
    timestamp: datetime
    acknowledged: bool = False
    source: str = "dashboard"

class BondXAIDashboard:
    """Enhanced real-time dashboard for BondX AI training monitoring with enterprise features"""
    
    def __init__(self, output_dir: str = "autonomous_training_output"):
        """Initialize the enhanced dashboard"""
        self.output_dir = Path(output_dir)
        self.dashboard_data = {}
        self.last_update = None
        self.update_interval = 5  # seconds
        
        # Dashboard state
        self.is_running = False
        self.start_time = None
        
        # Enhanced performance tracking
        self.performance_history = []
        self.quality_history = []
        self.convergence_history = []
        self.system_metrics_history = []
        self.alert_history = []
        
        # Enterprise monitoring configuration
        self.prometheus_enabled = False
        self.grafana_enabled = False
        self.alerting_enabled = False
        
        # Alert thresholds
        self.alert_thresholds = {
            'accuracy_drop': 0.88,
            'quality_score_drop': 0.90,
            'convergence_stall_hours': 2.0,
            'system_cpu_high': 80.0,
            'system_memory_high': 85.0,
            'system_disk_high': 90.0
        }
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"Enhanced BondX AI Dashboard initialized for directory: {output_dir}")
    
    async def start_monitoring(self):
        """Start the enhanced real-time monitoring dashboard"""
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("Starting Enhanced BondX AI training dashboard...")
        print("\n" + "="*80)
        print("🚀 ENHANCED BONDX AI AUTONOMOUS TRAINING DASHBOARD")
        print("="*80)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Enhanced monitoring with production-grade features...")
        print("="*80)
        
        try:
            # Start background monitoring
            self._start_background_monitoring()
            
            while self.is_running:
                await self._update_dashboard()
                await self._display_dashboard()
                await self._check_alerts()
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            self.is_running = False
        finally:
            self._stop_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background monitoring threads"""
        # System metrics monitoring
        self.system_monitor_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
        self.system_monitor_thread.start()
        
        # Performance monitoring
        self.performance_monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.performance_monitor_thread.start()
        
        logger.info("Background monitoring threads started")
    
    def _stop_background_monitoring(self):
        """Stop background monitoring threads"""
        self.is_running = False
        logger.info("Background monitoring threads stopped")
    
    def _monitor_system_metrics(self):
        """Monitor system metrics in background thread"""
        while self.is_running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage_percent = disk.percent
                
                # Network I/O
                network_io = psutil.net_io_counters()
                network_data = {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                }
                
                system_metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    disk_usage_percent=disk_usage_percent,
                    network_io=network_data,
                    timestamp=datetime.now()
                )
                
                self.system_metrics_history.append(system_metrics)
                
                # Keep only last 1000 metrics
                if len(self.system_metrics_history) > 1000:
                    self.system_metrics_history = self.system_metrics_history[-1000:]
                
                # Check system alerts
                self._check_system_alerts(system_metrics)
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                time.sleep(10)
    
    def _monitor_performance(self):
        """Monitor performance metrics in background thread"""
        while self.is_running:
            try:
                # Monitor training performance
                if self.performance_history:
                    recent_performance = self.performance_history[-10:]  # Last 10 entries
                    
                    # Calculate performance trends
                    if len(recent_performance) >= 2:
                        performance_trend = self._calculate_performance_trend(recent_performance)
                        
                        # Check for performance degradation
                        if performance_trend < -0.05:  # 5% degradation
                            self._create_alert('warning', f"Performance degradation detected: {performance_trend:.2%}")
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
                time.sleep(15)
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alert conditions"""
        if metrics.cpu_percent > self.alert_thresholds['system_cpu_high']:
            self._create_alert('warning', f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['system_memory_high']:
            self._create_alert('warning', f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > self.alert_thresholds['system_disk_high']:
            self._create_alert('warning', f"High disk usage: {metrics.disk_usage_percent:.1f}%")
    
    def _create_alert(self, level: str, message: str):
        """Create a new alert"""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            source="dashboard"
        )
        
        self.alert_history.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Log alert
        logger.warning(f"ALERT [{level.upper()}]: {message}")
        
        # TODO: Send to external alerting systems (Slack, email, etc.)
    
    def _calculate_performance_trend(self, performance_data: List[Dict]) -> float:
        """Calculate performance trend over recent data"""
        if len(performance_data) < 2:
            return 0.0
        
        # Extract quality scores
        quality_scores = [p.get('quality_score', 0) for p in performance_data]
        
        # Calculate trend (positive = improving, negative = degrading)
        if len(quality_scores) >= 2:
            trend = (quality_scores[-1] - quality_scores[0]) / quality_scores[0]
            return trend
        
        return 0.0
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        try:
            # Check training metrics for alerts
            if self.performance_history:
                latest_performance = self.performance_history[-1]
                
                # Check accuracy drop
                if 'accuracy' in latest_performance:
                    accuracy = latest_performance['accuracy']
                    if accuracy < self.alert_thresholds['accuracy_drop']:
                        self._create_alert('error', f"Model accuracy dropped below threshold: {accuracy:.3f}")
                
                # Check quality score drop
                if 'quality_score' in latest_performance:
                    quality_score = latest_performance['quality_score']
                    if quality_score < self.alert_thresholds['quality_score_drop']:
                        self._create_alert('error', f"Quality score dropped below threshold: {quality_score:.3f}")
            
            # Check convergence stall
            if self.convergence_history:
                latest_convergence = self.convergence_history[-1]
                if 'epochs_since_improvement' in latest_convergence:
                    epochs_stalled = latest_convergence['epochs_since_improvement']
                    if epochs_stalled > 0:
                        stall_hours = epochs_stalled * 0.1  # Approximate
                        if stall_hours >= self.alert_thresholds['convergence_stall_hours']:
                            self._create_alert('warning', f"Training stalled for {stall_hours:.1f} hours")
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _update_dashboard(self):
        """Update dashboard data from training output files"""
        try:
            # Check for new epoch reports
            epoch_reports = list(self.output_dir.glob("epoch_*_report.json"))
            if epoch_reports:
                latest_epoch = max(epoch_reports, key=lambda x: int(x.stem.split('_')[1]))
                await self._load_epoch_report(latest_epoch)
            
            # Check for dashboard metrics
            dashboard_files = list(self.output_dir.glob("epoch_*_dashboard.json"))
            if dashboard_files:
                latest_dashboard = max(dashboard_files, key=lambda x: int(x.stem.split('_')[1]))
                await self._load_dashboard_metrics(latest_dashboard)
            
            # Check for final report
            final_report = self.output_dir / "final_training_report.json"
            if final_report.exists():
                await self._load_final_report(final_report)
            
            # Check for stress test results
            stress_test_files = list(self.output_dir.glob("epoch_*_stress_tests.json"))
            if stress_test_files:
                latest_stress = max(stress_test_files, key=lambda x: int(x.stem.split('_')[1]))
                await self._load_stress_test_results(latest_stress)
            
            # Update performance history
            self._update_performance_history()
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    async def _load_epoch_report(self, report_path: Path):
        """Load epoch report data"""
        try:
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            epoch_num = report_data.get('epoch', 0)
            self.dashboard_data[f'epoch_{epoch_num}'] = report_data
            
            logger.debug(f"Loaded epoch {epoch_num} report")
            
        except Exception as e:
            logger.warning(f"Error loading epoch report {report_path}: {e}")
    
    async def _load_dashboard_metrics(self, dashboard_path: Path):
        """Load dashboard metrics"""
        try:
            with open(dashboard_path, 'r') as f:
                metrics_data = json.load(f)
            
            epoch_num = metrics_data.get('current_epoch', 0)
            self.dashboard_data[f'dashboard_{epoch_num}'] = metrics_data
            
            logger.debug(f"Loaded dashboard metrics for epoch {epoch_num}")
            
        except Exception as e:
            logger.warning(f"Error loading dashboard metrics {dashboard_path}: {e}")
    
    async def _load_final_report(self, report_path: Path):
        """Load final training report"""
        try:
            with open(report_path, 'r') as f:
                final_data = json.load(f)
            
            self.dashboard_data['final_report'] = final_data
            logger.info("Final training report loaded")
            
        except Exception as e:
            logger.warning(f"Error loading final report: {e}")
    
    async def _load_stress_test_results(self, stress_file: Path):
        """Load stress test results"""
        try:
            with open(stress_file, 'r') as f:
                stress_data = json.load(f)
            
            # Extract epoch number
            epoch_num = int(stress_file.stem.split('_')[1])
            
            # Store stress test results
            self.dashboard_data[f'stress_tests_epoch_{epoch_num}'] = stress_data
            
            logger.info(f"Loaded stress test results for epoch {epoch_num}")
            
        except Exception as e:
            logger.error(f"Error loading stress test results: {e}")
    
    def _update_performance_history(self):
        """Update performance tracking history"""
        current_time = datetime.now()
        
        # Extract current performance metrics
        current_metrics = self._extract_current_metrics()
        if current_metrics:
            self.performance_history.append({
                'timestamp': current_time,
                **current_metrics
            })
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def _extract_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Extract current performance metrics from dashboard data"""
        if not self.dashboard_data:
            return None
        
        # Find latest epoch data
        epoch_keys = [k for k in self.dashboard_data.keys() if k.startswith('epoch_')]
        if not epoch_keys:
            return None
        
        latest_epoch = max(epoch_keys, key=lambda x: int(x.split('_')[1]))
        epoch_data = self.dashboard_data[latest_epoch]
        
        # Extract metrics
        training_metrics = epoch_data.get('training_metrics', [])
        if not training_metrics:
            return None
        
        # Calculate averages
        mse_values = [m.get('mse', 0) for m in training_metrics]
        accuracy_values = [m.get('accuracy', 0) for m in training_metrics]
        
        return {
            'epoch': epoch_data.get('epoch', 0),
            'avg_mse': np.mean(mse_values) if mse_values else 0,
            'avg_accuracy': np.mean(accuracy_values) if accuracy_values else 0,
            'models_trained': len(training_metrics)
        }
    
    async def _display_dashboard(self):
        """Display the enhanced dashboard"""
        try:
            # Clear screen (platform independent)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Display header
            self._display_header()
            
            # Display training progress
            self._display_training_progress()
            
            # Display quality metrics
            self._display_quality_metrics()
            
            # Display convergence status
            self._display_convergence_status()
            
            # Display system metrics
            self._display_system_metrics()
            
            # Display recent alerts
            self._display_alerts()
            
            # Display stress test results
            self._display_stress_test_results()
            
            # Display footer
            self._display_footer()
            
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
    
    def _display_header(self):
        """Display dashboard header"""
        print("\n" + "="*80)
        print("🚀 ENHANCED BONDX AI AUTONOMOUS TRAINING DASHBOARD")
        print("="*80)
        print(f"Last Update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never'}")
        print(f"Running Time: {self._get_running_time()}")
        print("="*80)
    
    def _display_training_progress(self):
        """Display training progress section"""
        print("\n📊 TRAINING PROGRESS")
        print("-" * 40)
        
        if self.performance_history:
            latest = self.performance_history[-1]
            print(f"Current Epoch: {latest.get('epoch', 'N/A')}")
            print(f"Total Epochs: {len(self.performance_history)}")
            print(f"Training Time: {latest.get('training_time', 'N/A')}")
        else:
            print("No training data available")
    
    def _display_quality_metrics(self):
        """Display quality metrics section"""
        print("\n🔍 QUALITY METRICS")
        print("-" * 40)
        
        if self.quality_history:
            latest = self.quality_history[-1]
            print(f"Quality Score: {latest.get('quality_score', 'N/A'):.3f}")
            print(f"Coverage: {latest.get('coverage', 'N/A'):.1f}%")
            print(f"ESG Completeness: {latest.get('esg_completeness', 'N/A'):.1f}%")
            print(f"Liquidity Median: {latest.get('liquidity_median', 'N/A'):.1f}")
        else:
            print("No quality data available")
    
    def _display_convergence_status(self):
        """Display convergence status section"""
        print("\n🎯 CONVERGENCE STATUS")
        print("-" * 40)
        
        if self.convergence_history:
            latest = self.convergence_history[-1]
            print(f"Models Converged: {latest.get('models_converged', 'N/A')}")
            print(f"Quality Stable: {latest.get('quality_stable', 'N/A')}")
            print(f"Improvement Rate: {latest.get('improvement_rate', 'N/A'):.6f}")
            print(f"Epochs Since Improvement: {latest.get('epochs_since_improvement', 'N/A')}")
        else:
            print("No convergence data available")
    
    def _display_system_metrics(self):
        """Display system metrics section"""
        print("\n💻 SYSTEM METRICS")
        print("-" * 40)
        
        if self.system_metrics_history:
            latest = self.system_metrics_history[-1]
            print(f"CPU Usage: {latest.cpu_percent:.1f}%")
            print(f"Memory Usage: {latest.memory_percent:.1f}%")
            print(f"Disk Usage: {latest.disk_usage_percent:.1f}%")
            print(f"Network Sent: {latest.network_io['bytes_sent'] / 1024 / 1024:.1f} MB")
            print(f"Network Received: {latest.network_io['bytes_recv'] / 1024 / 1024:.1f} MB")
        else:
            print("No system metrics available")
    
    def _display_alerts(self):
        """Display recent alerts section"""
        print("\n🚨 RECENT ALERTS")
        print("-" * 40)
        
        if self.alert_history:
            recent_alerts = self.alert_history[-5:]  # Show last 5 alerts
            for alert in recent_alerts:
                status = "🔴" if alert.level in ['error', 'critical'] else "🟡" if alert.level == 'warning' else "🔵"
                print(f"{status} [{alert.level.upper()}] {alert.message}")
                print(f"   {alert.timestamp.strftime('%H:%M:%S')}")
        else:
            print("No alerts")
    
    def _display_stress_test_results(self):
        """Display stress test results section"""
        print("\n🧪 STRESS TEST RESULTS")
        print("-" * 40)
        
        stress_data = {k: v for k, v in self.dashboard_data.items() if 'stress_tests' in k}
        if stress_data:
            latest_stress = max(stress_data.keys(), key=lambda x: int(x.split('_')[2]))
            stress_results = stress_data[latest_stress]
            
            for scenario, result in stress_results.items():
                if isinstance(result, dict) and 'liquidity_deterioration' in result:
                    deterioration = result['liquidity_deterioration']
                    status = "🔴" if deterioration > 20 else "🟡" if deterioration > 10 else "🟢"
                    print(f"{status} {scenario}: {deterioration:.1f}% deterioration")
        else:
            print("No stress test results available")
    
    def _display_footer(self):
        """Display dashboard footer"""
        print("\n" + "="*80)
        print("Press Ctrl+C to stop monitoring")
        print("="*80)
    
    def _get_running_time(self) -> str:
        """Get formatted running time"""
        if not self.start_time:
            return "N/A"
        
        elapsed = datetime.now() - self.start_time
        hours = elapsed.total_seconds() // 3600
        minutes = (elapsed.total_seconds() % 3600) // 60
        seconds = elapsed.total_seconds() % 60
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.metrics = []
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics.append({
            'name': name,
            'value': value,
            'timestamp': timestamp
        })
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def get_metric_trend(self, name: str, window_minutes: int = 60) -> float:
        """Get trend for a specific metric over time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_metrics = [
            m for m in self.metrics 
            if m['name'] == name and m['timestamp'] > cutoff_time
        ]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate trend
        values = [m['value'] for m in recent_metrics]
        if values[0] != 0:
            trend = (values[-1] - values[0]) / values[0]
            return trend
        
        return 0.0

async def main():
    """Main entry point for the dashboard"""
    # Create dashboard
    dashboard = BondXAIDashboard()
    
    try:
        # Start monitoring
        await dashboard.start_monitoring()
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    finally:
        # Generate final summary
        if dashboard.dashboard_data:
            print("\n📋 GENERATING FINAL SUMMARY...")
            summary = await dashboard.generate_summary_report()
            
            # Save summary
            summary_path = dashboard.output_dir / "dashboard_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📄 Summary saved to: {summary_path}")
        
        print("\n👋 Dashboard session ended")

if __name__ == "__main__":
    # Run the dashboard
    asyncio.run(main())
