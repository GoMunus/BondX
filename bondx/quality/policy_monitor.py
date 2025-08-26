#!/usr/bin/env python3
"""
Policy Drift Detection for BondX Quality Assurance

Monitors quality policy effectiveness over time to detect when policies
effectively loosen without explicit policy version changes.

Detects patterns like:
- FAIL/WARN rates dropping significantly
- Coverage thresholds being exceeded more frequently
- ESG/liquidity gates becoming more permissive
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

class PolicyDriftDetector:
    """Detects policy drift in quality assurance outcomes."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the policy drift detector."""
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Default thresholds if config not available
        self.default_thresholds = {
            'fail_rate_drop_threshold': 0.15,  # 15% drop in FAIL rate
            'warn_rate_drop_threshold': 0.20,  # 20% drop in WARN rate
            'coverage_increase_threshold': 0.10,  # 10% increase in coverage
            'esg_completeness_increase_threshold': 0.15,  # 15% increase in ESG completeness
            'liquidity_improvement_threshold': 0.20,  # 20% improvement in liquidity
            'moving_average_window': 30,  # 30-day moving average
            'minimum_data_points': 7,  # Minimum data points for analysis
            'confidence_level': 0.95  # 95% confidence for drift detection
        }
        
        # Merge with config if available
        if self.config:
            self.thresholds = {**self.default_thresholds, **self.config.get('thresholds', {})}
        else:
            self.thresholds = self.default_thresholds
    
    def _load_config(self, config_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        if not config_path:
            config_path = "bondx/quality/config/policy_drift.yaml"
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, ImportError, yaml.YAMLError) as e:
            logging.warning(f"Could not load policy drift config from {config_path}: {e}")
            return None
    
    def load_quality_history(self, history_dir: str = "bondx/quality/reports/history") -> pd.DataFrame:
        """Load quality assurance history from JSON snapshots."""
        history_path = Path(history_dir)
        
        if not history_path.exists():
            self.logger.warning(f"History directory not found: {history_path}")
            return pd.DataFrame()
        
        # Find all JSON snapshot files
        snapshot_files = list(history_path.glob("*.json"))
        
        if not snapshot_files:
            self.logger.warning(f"No snapshot files found in {history_path}")
            return pd.DataFrame()
        
        # Load and combine snapshots
        snapshots = []
        for snapshot_file in sorted(snapshot_files):
            try:
                with open(snapshot_file, 'r') as f:
                    snapshot = json.load(f)
                
                # Extract timestamp from filename or content
                timestamp = self._extract_timestamp(snapshot_file, snapshot)
                
                # Extract key metrics
                metrics = self._extract_metrics(snapshot)
                if metrics:
                    metrics['timestamp'] = timestamp
                    snapshots.append(metrics)
                    
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Could not load snapshot {snapshot_file}: {e}")
                continue
        
        if not snapshots:
            return pd.DataFrame()
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(snapshots)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"Loaded {len(df)} quality snapshots from {history_path}")
        return df
    
    def _extract_timestamp(self, filename: Path, snapshot: Dict[str, Any]) -> pd.Timestamp:
        """Extract timestamp from filename or snapshot content."""
        # Try to get timestamp from filename first
        try:
            # Look for timestamp patterns in filename
            import re
            timestamp_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
                r'(\d{8})',  # YYYYMMDD
            ]
            
            for pattern in timestamp_patterns:
                match = re.search(pattern, filename.name)
                if match:
                    date_str = match.group(1)
                    if '_' in date_str:
                        date_str = date_str.replace('_', '-')
                    return pd.to_datetime(date_str)
        except:
            pass
        
        # Fall back to snapshot content
        try:
            if 'timestamp' in snapshot:
                return pd.to_datetime(snapshot['timestamp'])
            elif 'run_timestamp' in snapshot:
                return pd.to_datetime(snapshot['run_timestamp'])
            elif 'generated_at' in snapshot:
                return pd.to_datetime(snapshot['generated_at'])
        except:
            pass
        
        # Default to file modification time
        return pd.to_datetime(filename.stat().st_mtime, unit='s')
    
    def _extract_metrics(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract key metrics from quality snapshot."""
        try:
            metrics = {}
            
            # Extract validation results
            if 'validation_results' in snapshot:
                validation_results = snapshot['validation_results']
                if isinstance(validation_results, list):
                    # Count by severity
                    severity_counts = defaultdict(int)
                    for result in validation_results:
                        if isinstance(result, dict) and 'severity' in result:
                            severity = result['severity']
                            severity_counts[severity] += 1
                    
                    total_validations = sum(severity_counts.values())
                    if total_validations > 0:
                        metrics['total_validations'] = total_validations
                        metrics['pass_count'] = severity_counts.get('PASS', 0)
                        metrics['warn_count'] = severity_counts.get('WARN', 0)
                        metrics['fail_count'] = severity_counts.get('FAIL', 0)
                        metrics['pass_rate'] = severity_counts.get('PASS', 0) / total_validations
                        metrics['warn_rate'] = severity_counts.get('WARN', 0) / total_validations
                        metrics['fail_rate'] = severity_counts.get('FAIL', 0) / total_validations
            
            # Extract quality gate results
            if 'gate_results' in snapshot:
                gate_results = snapshot['gate_results']
                if isinstance(gate_results, dict):
                    gate_counts = defaultdict(int)
                    for gate_name, gate_result in gate_results.items():
                        if isinstance(gate_result, dict) and 'status' in gate_result:
                            status = gate_result['status']
                            gate_counts[status] += 1
                    
                    total_gates = sum(gate_counts.values())
                    if total_gates > 0:
                        metrics['total_gates'] = total_gates
                        metrics['gate_pass_count'] = gate_counts.get('PASS', 0)
                        metrics['gate_warn_count'] = gate_counts.get('WARN', 0)
                        metrics['gate_fail_count'] = gate_counts.get('FAIL', 0)
                        metrics['gate_pass_rate'] = gate_counts.get('PASS', 0) / total_gates
                        metrics['gate_warn_rate'] = gate_counts.get('WARN', 0) / total_gates
                        metrics['gate_fail_rate'] = gate_counts.get('FAIL', 0) / total_gates
            
            # Extract key metrics
            if 'metrics' in snapshot:
                snapshot_metrics = snapshot['metrics']
                if isinstance(snapshot_metrics, dict):
                    for key in ['coverage_percent', 'esg_completeness_percent', 
                               'liquidity_index_median', 'data_freshness_minutes']:
                        if key in snapshot_metrics:
                            metrics[key] = snapshot_metrics[key]
            
            return metrics if metrics else None
            
        except Exception as e:
            self.logger.warning(f"Error extracting metrics from snapshot: {e}")
            return None
    
    def detect_drift(self, history_df: pd.DataFrame, 
                     policy_version: Optional[str] = None) -> Dict[str, Any]:
        """Detect policy drift in quality metrics."""
        if history_df.empty:
            return {
                'drift_detected': False,
                'message': 'No historical data available',
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        # Ensure we have enough data
        if len(history_df) < self.thresholds['minimum_data_points']:
            return {
                'drift_detected': False,
                'message': f'Insufficient data points: {len(history_df)} < {self.thresholds["minimum_data_points"]}',
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        # Calculate moving averages
        window = self.thresholds['moving_average_window']
        if len(history_df) < window:
            window = len(history_df) // 2
        
        drift_indicators = []
        
        # Check for FAIL rate drops
        if 'fail_rate' in history_df.columns:
            fail_rate_ma = history_df['fail_rate'].rolling(window=window).mean()
            if len(fail_rate_ma.dropna()) > 0:
                recent_fail_rate = fail_rate_ma.iloc[-1]
                baseline_fail_rate = fail_rate_ma.iloc[:window//2].mean()
                
                if baseline_fail_rate > 0:
                    fail_rate_change = (baseline_fail_rate - recent_fail_rate) / baseline_fail_rate
                    
                    if fail_rate_change > self.thresholds['fail_rate_drop_threshold']:
                        drift_indicators.append({
                            'metric': 'fail_rate',
                            'change': fail_rate_change,
                            'threshold': self.thresholds['fail_rate_drop_threshold'],
                            'baseline': baseline_fail_rate,
                            'recent': recent_fail_rate,
                            'severity': 'high' if fail_rate_change > 0.3 else 'medium'
                        })
        
        # Check for WARN rate drops
        if 'warn_rate' in history_df.columns:
            warn_rate_ma = history_df['warn_rate'].rolling(window=window).mean()
            if len(warn_rate_ma.dropna()) > 0:
                recent_warn_rate = warn_rate_ma.iloc[-1]
                baseline_warn_rate = warn_rate_ma.iloc[:window//2].mean()
                
                if baseline_warn_rate > 0:
                    warn_rate_change = (baseline_warn_rate - recent_warn_rate) / baseline_warn_rate
                    
                    if warn_rate_change > self.thresholds['warn_rate_drop_threshold']:
                        drift_indicators.append({
                            'metric': 'warn_rate',
                            'change': warn_rate_change,
                            'threshold': self.thresholds['warn_rate_drop_threshold'],
                            'baseline': baseline_warn_rate,
                            'recent': recent_warn_rate,
                            'severity': 'high' if warn_rate_change > 0.4 else 'medium'
                        })
        
        # Check for coverage increases
        if 'coverage_percent' in history_df.columns:
            coverage_ma = history_df['coverage_percent'].rolling(window=window).mean()
            if len(coverage_ma.dropna()) > 0:
                recent_coverage = coverage_ma.iloc[-1]
                baseline_coverage = coverage_ma.iloc[:window//2].mean()
                
                coverage_change = (recent_coverage - baseline_coverage) / baseline_coverage
                
                if coverage_change > self.thresholds['coverage_increase_threshold']:
                    drift_indicators.append({
                        'metric': 'coverage_percent',
                        'change': coverage_change,
                        'threshold': self.thresholds['coverage_increase_threshold'],
                        'baseline': baseline_coverage,
                        'recent': recent_coverage,
                        'severity': 'medium'
                    })
        
        # Check for ESG completeness improvements
        if 'esg_completeness_percent' in history_df.columns:
            esg_ma = history_df['esg_completeness_percent'].rolling(window=window).mean()
            if len(esg_ma.dropna()) > 0:
                recent_esg = esg_ma.iloc[-1]
                baseline_esg = esg_ma.iloc[:window//2].mean()
                
                if baseline_esg > 0:
                    esg_change = (recent_esg - baseline_esg) / baseline_esg
                    
                    if esg_change > self.thresholds['esg_completeness_increase_threshold']:
                        drift_indicators.append({
                            'metric': 'esg_completeness_percent',
                            'change': esg_change,
                            'threshold': self.thresholds['esg_completeness_increase_threshold'],
                            'baseline': baseline_esg,
                            'recent': recent_esg,
                            'severity': 'medium'
                        })
        
        # Determine overall drift status
        drift_detected = len(drift_indicators) > 0
        
        # Check if policy version has changed
        policy_changed = self._check_policy_version_change(policy_version)
        
        # Generate drift report
        drift_report = {
            'drift_detected': drift_detected,
            'policy_version_changed': policy_changed,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_points_analyzed': len(history_df),
            'moving_average_window': window,
            'drift_indicators': drift_indicators,
            'thresholds_used': self.thresholds,
            'recommendations': self._generate_recommendations(drift_indicators, policy_changed)
        }
        
        if drift_detected and not policy_changed:
            drift_report['severity'] = 'high'
            drift_report['message'] = 'Policy drift detected without version change'
        elif drift_detected and policy_changed:
            drift_report['severity'] = 'medium'
            drift_report['message'] = 'Policy drift detected with version change'
        else:
            drift_report['severity'] = 'low'
            drift_report['message'] = 'No significant policy drift detected'
        
        return drift_report
    
    def _check_policy_version_change(self, current_policy_version: Optional[str]) -> bool:
        """Check if policy version has changed recently."""
        if not current_policy_version:
            return False
        
        # Check policy version file
        policy_version_file = Path("bondx/quality/policy_version.txt")
        if policy_version_file.exists():
            try:
                with open(policy_version_file, 'r') as f:
                    stored_version = f.read().strip()
                
                return stored_version != current_policy_version
            except Exception as e:
                self.logger.warning(f"Could not read policy version: {e}")
        
        return False
    
    def _generate_recommendations(self, drift_indicators: List[Dict], 
                                 policy_changed: bool) -> List[str]:
        """Generate recommendations based on drift indicators."""
        recommendations = []
        
        if not drift_indicators:
            recommendations.append("Continue monitoring quality metrics")
            return recommendations
        
        if not policy_changed:
            recommendations.append("Review quality policy implementation for unintended changes")
            recommendations.append("Check for data quality improvements that may mask policy issues")
            recommendations.append("Consider tightening quality thresholds if improvements are sustainable")
        
        for indicator in drift_indicators:
            metric = indicator['metric']
            change = indicator['change']
            severity = indicator['severity']
            
            if metric == 'fail_rate':
                if severity == 'high':
                    recommendations.append(f"Investigate significant drop in FAIL rate ({change:.1%})")
                recommendations.append("Verify that quality standards are being maintained")
                recommendations.append("Check for changes in data sources or validation logic")
            
            elif metric == 'warn_rate':
                recommendations.append(f"Review WARN rate reduction ({change:.1%}) for appropriateness")
                recommendations.append("Ensure warnings are still being generated for borderline cases")
            
            elif metric == 'coverage_percent':
                recommendations.append(f"Coverage improvement ({change:.1%}) may indicate policy relaxation")
                recommendations.append("Verify that coverage thresholds remain appropriate")
            
            elif metric == 'esg_completeness_percent':
                recommendations.append(f"ESG completeness improvement ({change:.1%}) - verify data quality")
                recommendations.append("Check for changes in ESG data collection or validation")
        
        recommendations.append("Update policy version if changes are intentional")
        recommendations.append("Document rationale for any policy adjustments")
        
        return recommendations
    
    def export_metrics_csv(self, history_df: pd.DataFrame, 
                          output_path: str = "quality/policy_drift_metrics.csv") -> str:
        """Export quality metrics to CSV for dashboard import."""
        if history_df.empty:
            return ""
        
        # Prepare metrics for export
        export_df = history_df.copy()
        
        # Add moving averages
        window = self.thresholds['moving_average_window']
        if len(export_df) >= window:
            for col in ['fail_rate', 'warn_rate', 'pass_rate']:
                if col in export_df.columns:
                    export_df[f'{col}_ma'] = export_df[col].rolling(window=window).mean()
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        export_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Metrics exported to CSV: {output_path}")
        return str(output_path)
    
    def generate_human_summary(self, drift_report: Dict[str, Any]) -> str:
        """Generate human-readable summary of drift analysis."""
        summary = f"""Policy Drift Analysis Summary
Generated: {drift_report['analysis_timestamp']}

OVERALL STATUS: {drift_report['message']}
Severity: {drift_report['severity'].upper()}

ANALYSIS DETAILS:
- Data points analyzed: {drift_report['data_points_analyzed']}
- Moving average window: {drift_report['moving_average_window']} days
- Policy version changed: {'Yes' if drift_report['policy_version_changed'] else 'No'}

DRIFT INDICATORS:
"""
        
        if drift_report['drift_indicators']:
            for indicator in drift_report['drift_indicators']:
                summary += f"- {indicator['metric']}: {indicator['change']:.1%} change "
                summary += f"({indicator['severity']} severity)\n"
        else:
            summary += "- None detected\n"
        
        summary += f"\nRECOMMENDATIONS:\n"
        for rec in drift_report['recommendations']:
            summary += f"- {rec}\n"
        
        return summary

def main():
    """Main function for policy drift detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect policy drift in quality assurance")
    parser.add_argument('--history-dir', default='bondx/quality/reports/history',
                       help='Directory containing quality history snapshots')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', default='quality/policy_drift.json',
                       help='Output file for drift report')
    parser.add_argument('--csv', default='quality/policy_drift_metrics.csv',
                       help='Output CSV for metrics export')
    parser.add_argument('--summary', action='store_true',
                       help='Print human-readable summary')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize detector
    detector = PolicyDriftDetector(args.config)
    
    # Load history
    history_df = detector.load_quality_history(args.history_dir)
    
    if history_df.empty:
        print("No quality history data found. Exiting.")
        return 1
    
    # Detect drift
    drift_report = detector.detect_drift(history_df)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(drift_report, f, indent=2)
    
    print(f"Drift report saved to: {output_path}")
    
    # Export metrics CSV
    if args.csv:
        csv_path = detector.export_metrics_csv(history_df, args.csv)
        if csv_path:
            print(f"Metrics exported to: {csv_path}")
    
    # Print summary
    if args.summary:
        summary = detector.generate_human_summary(drift_report)
        print("\n" + "="*60)
        print(summary)
        print("="*60)
    
    # Exit with appropriate code
    if drift_report['drift_detected'] and not drift_report['policy_version_changed']:
        print("\n❌ POLICY DRIFT DETECTED - Review required!")
        return 1
    else:
        print("\n✅ No significant policy drift detected")
        return 0

if __name__ == "__main__":
    exit(main())
