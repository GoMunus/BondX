#!/usr/bin/env python3
"""
Quality Metrics Exporter for Grafana Dashboards

Exports quality gate outcomes and metrics in Prometheus-friendly format
for integration with Grafana dashboards.

Usage:
    python -m bondx.quality.metrics_exporter --input quality/last_run_report.json --out quality/metrics
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

class QualityMetricsExporter:
    """Exports quality metrics in Prometheus format for Grafana."""
    
    def __init__(self):
        """Initialize the metrics exporter."""
        self.logger = logging.getLogger(__name__)
        
        # Prometheus metric types
        self.metric_types = {
            'counter': 'bondx_quality_gate_total',
            'gauge': 'bondx_quality_gate_current',
            'histogram': 'bondx_quality_gate_duration'
        }
    
    def load_quality_report(self, report_path: str) -> Dict[str, Any]:
        """Load quality report from JSON file."""
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            self.logger.info(f"Loaded quality report: {report_path}")
            return report
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Could not load quality report {report_path}: {e}")
            raise
    
    def extract_validation_metrics(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """Extract validation metrics from results."""
        metrics = {}
        
        # Count by severity
        severity_counts = {}
        rule_counts = {}
        sector_counts = {}
        
        for result in validation_results:
            severity = result.get('severity', 'UNKNOWN')
            rule_name = result.get('rule_name', 'unknown')
            sector = result.get('sector', 'unknown')
            dataset = result.get('dataset', 'unknown')
            
            # Count by severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by rule
            if rule_name not in rule_counts:
                rule_counts[rule_name] = {'total': 0, 'pass': 0, 'warn': 0, 'fail': 0}
            
            rule_counts[rule_name]['total'] += 1
            if severity in rule_counts[rule_name]:
                rule_counts[rule_name][severity.lower()] += 1
            
            # Count by sector
            if sector not in sector_counts:
                sector_counts[sector] = {'total': 0, 'pass': 0, 'warn': 0, 'fail': 0}
            
            sector_counts[sector]['total'] += 1
            if severity in sector_counts[sector]:
                sector_counts[sector][severity.lower()] += 1
        
        metrics['severity_counts'] = severity_counts
        metrics['rule_counts'] = rule_counts
        metrics['sector_counts'] = sector_counts
        
        # Calculate rates
        total_validations = len(validation_results)
        if total_validations > 0:
            metrics['pass_rate'] = severity_counts.get('PASS', 0) / total_validations
            metrics['warn_rate'] = severity_counts.get('WARN', 0) / total_validations
            metrics['fail_rate'] = severity_counts.get('FAIL', 0) / total_validations
        else:
            metrics['pass_rate'] = 0
            metrics['warn_rate'] = 0
            metrics['fail_rate'] = 0
        
        metrics['total_validations'] = total_validations
        
        return metrics
    
    def extract_gate_metrics(self, gate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality gate metrics from results."""
        metrics = {}
        
        # Count gate outcomes
        gate_counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
        gate_details = {}
        
        for gate_name, gate_result in gate_results.items():
            status = gate_result.get('status', 'UNKNOWN')
            if status in gate_counts:
                gate_counts[status] += 1
            
            # Store gate details
            gate_details[gate_name] = {
                'status': status,
                'threshold': gate_result.get('threshold', 0),
                'actual': gate_result.get('actual', 0),
                'message': gate_result.get('message', '')
            }
        
        metrics['gate_counts'] = gate_counts
        metrics['gate_details'] = gate_details
        
        # Calculate gate rates
        total_gates = len(gate_results)
        if total_gates > 0:
            metrics['gate_pass_rate'] = gate_counts['PASS'] / total_gates
            metrics['gate_warn_rate'] = gate_counts['WARN'] / total_gates
            metrics['gate_fail_rate'] = gate_counts['FAIL'] / total_gates
        else:
            metrics['gate_pass_rate'] = 0
            metrics['gate_warn_rate'] = 0
            metrics['gate_fail_rate'] = 0
        
        metrics['total_gates'] = total_gates
        
        return metrics
    
    def generate_prometheus_metrics(self, validation_metrics: Dict[str, Any], 
                                   gate_metrics: Dict[str, Any],
                                   additional_metrics: Dict[str, Any]) -> str:
        """Generate Prometheus-formatted metrics."""
        timestamp = int(datetime.now().timestamp() * 1000)
        
        prometheus_metrics = []
        
        # Add timestamp comment
        prometheus_metrics.append(f"# Quality metrics generated at {datetime.now().isoformat()}")
        prometheus_metrics.append("")
        
        # Validation severity metrics
        prometheus_metrics.append("# Validation severity metrics")
        for severity, count in validation_metrics['severity_counts'].items():
            metric_name = f"bondx_quality_validation_{severity.lower()}_total"
            prometheus_metrics.append(f'{metric_name}{{severity="{severity}"}} {count}')
        
        # Validation rates
        prometheus_metrics.append("")
        prometheus_metrics.append("# Validation rate metrics")
        prometheus_metrics.append(f'bondx_quality_pass_rate {validation_metrics["pass_rate"]:.4f}')
        prometheus_metrics.append(f'bondx_quality_warn_rate {validation_metrics["warn_rate"]:.4f}')
        prometheus_metrics.append(f'bondx_quality_fail_rate {validation_metrics["fail_rate"]:.4f}')
        
        # Rule-specific metrics
        prometheus_metrics.append("")
        prometheus_metrics.append("# Rule-specific validation metrics")
        for rule_name, counts in validation_metrics['rule_counts'].items():
            for severity in ['pass', 'warn', 'fail']:
                metric_name = f"bondx_quality_rule_{severity}_total"
                count = counts.get(severity, 0)
                prometheus_metrics.append(f'{metric_name}{{rule="{rule_name}",severity="{severity.upper()}"}} {count}')
        
        # Sector-specific metrics
        prometheus_metrics.append("")
        prometheus_metrics.append("# Sector-specific validation metrics")
        for sector, counts in validation_metrics['sector_counts'].items():
            for severity in ['pass', 'warn', 'fail']:
                metric_name = f"bondx_quality_sector_{severity}_total"
                count = counts.get(severity, 0)
                prometheus_metrics.append(f'{metric_name}{{sector="{sector}",severity="{severity.upper()}"}} {count}')
        
        # Quality gate metrics
        prometheus_metrics.append("")
        prometheus_metrics.append("# Quality gate metrics")
        for status, count in gate_metrics['gate_counts'].items():
            metric_name = f"bondx_quality_gate_{status.lower()}_total"
            prometheus_metrics.append(f'{metric_name}{{status="{status}"}} {count}')
        
        # Gate rates
        prometheus_metrics.append("")
        prometheus_metrics.append("# Quality gate rate metrics")
        prometheus_metrics.append(f'bondx_quality_gate_pass_rate {gate_metrics["gate_pass_rate"]:.4f}')
        prometheus_metrics.append(f'bondx_quality_gate_warn_rate {gate_metrics["gate_warn_rate"]:.4f}')
        prometheus_metrics.append(f'bondx_quality_gate_fail_rate {gate_metrics["gate_fail_rate"]:.4f}')
        
        # Additional metrics
        if additional_metrics:
            prometheus_metrics.append("")
            prometheus_metrics.append("# Additional quality metrics")
            
            for metric_name, value in additional_metrics.items():
                if isinstance(value, (int, float)):
                    # Convert metric name to Prometheus format
                    prom_name = f"bondx_quality_{metric_name.lower()}"
                    prometheus_metrics.append(f'{prom_name} {value}')
        
        # Total counts
        prometheus_metrics.append("")
        prometheus_metrics.append("# Total counts")
        prometheus_metrics.append(f'bondx_quality_total_validations {validation_metrics["total_validations"]}')
        prometheus_metrics.append(f'bondx_quality_total_gates {gate_metrics["total_gates"]}')
        
        return "\n".join(prometheus_metrics)
    
    def generate_csv_metrics(self, validation_metrics: Dict[str, Any], 
                            gate_metrics: Dict[str, Any],
                            additional_metrics: Dict[str, Any]) -> str:
        """Generate CSV format metrics for Grafana import."""
        timestamp = datetime.now().isoformat()
        
        # Prepare data for CSV
        csv_data = []
        
        # Validation metrics
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'validation',
            'metric_name': 'total_validations',
            'value': validation_metrics['total_validations'],
            'severity': 'N/A',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'validation',
            'metric_name': 'pass_rate',
            'value': validation_metrics['pass_rate'],
            'severity': 'PASS',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'validation',
            'metric_name': 'warn_rate',
            'value': validation_metrics['warn_rate'],
            'severity': 'WARN',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'validation',
            'metric_name': 'fail_rate',
            'value': validation_metrics['fail_rate'],
            'severity': 'FAIL',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        # Rule-specific metrics
        for rule_name, counts in validation_metrics['rule_counts'].items():
            for severity in ['pass', 'warn', 'fail']:
                csv_data.append({
                    'timestamp': timestamp,
                    'metric_type': 'rule_validation',
                    'metric_name': f'{severity}_count',
                    'value': counts.get(severity, 0),
                    'severity': severity.upper(),
                    'rule': rule_name,
                    'sector': 'N/A'
                })
        
        # Sector-specific metrics
        for sector, counts in validation_metrics['sector_counts'].items():
            for severity in ['pass', 'warn', 'fail']:
                csv_data.append({
                    'timestamp': timestamp,
                    'metric_type': 'sector_validation',
                    'metric_name': f'{severity}_count',
                    'value': counts.get(severity, 0),
                    'severity': severity.upper(),
                    'rule': 'N/A',
                    'sector': sector
                })
        
        # Gate metrics
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'gate',
            'metric_name': 'total_gates',
            'value': gate_metrics['total_gates'],
            'severity': 'N/A',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'gate',
            'metric_name': 'pass_rate',
            'value': gate_metrics['gate_pass_rate'],
            'severity': 'PASS',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'gate',
            'metric_name': 'warn_rate',
            'value': gate_metrics['gate_warn_rate'],
            'severity': 'WARN',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        csv_data.append({
            'timestamp': timestamp,
            'metric_type': 'gate',
            'metric_name': 'fail_rate',
            'value': gate_metrics['gate_fail_rate'],
            'severity': 'FAIL',
            'rule': 'N/A',
            'sector': 'N/A'
        })
        
        # Additional metrics
        if additional_metrics:
            for metric_name, value in additional_metrics.items():
                if isinstance(value, (int, float)):
                    csv_data.append({
                        'timestamp': timestamp,
                        'metric_type': 'additional',
                        'metric_name': metric_name,
                        'value': value,
                        'severity': 'N/A',
                        'rule': 'N/A',
                        'sector': 'N/A'
                    })
        
        # Convert to DataFrame and then CSV
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
    
    def export_metrics(self, report_path: str, output_dir: str = "quality/metrics",
                      formats: List[str] = None) -> Dict[str, str]:
        """Export quality metrics in multiple formats."""
        if formats is None:
            formats = ['prometheus', 'csv']
        
        # Load quality report
        report = self.load_quality_report(report_path)
        
        # Extract metrics
        validation_results = report.get('validation_results', [])
        gate_results = report.get('gate_results', {})
        additional_metrics = report.get('metrics', {})
        
        validation_metrics = self.extract_validation_metrics(validation_results)
        gate_metrics = self.extract_gate_metrics(gate_results)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported_files = {}
        
        # Export Prometheus format
        if 'prometheus' in formats:
            prometheus_content = self.generate_prometheus_metrics(
                validation_metrics, gate_metrics, additional_metrics
            )
            
            prometheus_file = output_path / f"quality_metrics_{timestamp}.prom"
            with open(prometheus_file, 'w') as f:
                f.write(prometheus_content)
            
            exported_files['prometheus'] = str(prometheus_file)
            self.logger.info(f"Prometheus metrics exported to: {prometheus_file}")
        
        # Export CSV format
        if 'csv' in formats:
            csv_content = self.generate_csv_metrics(
                validation_metrics, gate_metrics, additional_metrics
            )
            
            csv_file = output_path / f"quality_metrics_{timestamp}.csv"
            with open(csv_file, 'w') as f:
                f.write(csv_content)
            
            exported_files['csv'] = str(csv_file)
            self.logger.info(f"CSV metrics exported to: {csv_file}")
        
        # Export JSON summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'validation_metrics': validation_metrics,
            'gate_metrics': gate_metrics,
            'additional_metrics': additional_metrics,
            'exported_files': exported_files
        }
        
        summary_file = output_path / f"metrics_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        exported_files['summary'] = str(summary_file)
        self.logger.info(f"Metrics summary exported to: {summary_file}")
        
        return exported_files
    
    def generate_grafana_dashboard_json(self, output_dir: str = "quality/metrics") -> str:
        """Generate Grafana dashboard JSON configuration."""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "BondX Quality Gate Dashboard",
                "tags": ["bondx", "quality", "monitoring"],
                "style": "dark",
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Quality Gate Overview",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "bondx_quality_total_gates",
                                "legendFormat": "Total Gates"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 80}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Validation Results by Severity",
                        "type": "piechart",
                        "targets": [
                            {
                                "expr": "bondx_quality_validation_pass_total",
                                "legendFormat": "PASS"
                            },
                            {
                                "expr": "bondx_quality_validation_warn_total",
                                "legendFormat": "WARN"
                            },
                            {
                                "expr": "bondx_quality_validation_fail_total",
                                "legendFormat": "FAIL"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Quality Gate Status",
                        "type": "barchart",
                        "targets": [
                            {
                                "expr": "bondx_quality_gate_pass_total",
                                "legendFormat": "PASS"
                            },
                            {
                                "expr": "bondx_quality_gate_warn_total",
                                "legendFormat": "WARN"
                            },
                            {
                                "expr": "bondx_quality_gate_fail_total",
                                "legendFormat": "FAIL"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Pass Rate Trends",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": "bondx_quality_pass_rate",
                                "legendFormat": "Pass Rate"
                            },
                            {
                                "expr": "bondx_quality_warn_rate",
                                "legendFormat": "Warn Rate"
                            },
                            {
                                "expr": "bondx_quality_fail_rate",
                                "legendFormat": "Fail Rate"
                            }
                        ]
                    },
                    {
                        "id": 5,
                        "title": "Validation Results by Rule",
                        "type": "table",
                        "targets": [
                            {
                                "expr": "bondx_quality_rule_pass_total",
                                "legendFormat": "{{rule}} - PASS"
                            },
                            {
                                "expr": "bondx_quality_rule_warn_total",
                                "legendFormat": "{{rule}} - WARN"
                            },
                            {
                                "expr": "bondx_quality_rule_fail_total",
                                "legendFormat": "{{rule}} - FAIL"
                            }
                        ]
                    },
                    {
                        "id": 6,
                        "title": "Sector Quality Overview",
                        "type": "heatmap",
                        "targets": [
                            {
                                "expr": "bondx_quality_sector_pass_total",
                                "legendFormat": "{{sector}} - PASS"
                            },
                            {
                                "expr": "bondx_quality_sector_warn_total",
                                "legendFormat": "{{sector}} - WARN"
                            },
                            {
                                "expr": "bondx_quality_sector_fail_total",
                                "legendFormat": "{{sector}} - FAIL"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-24h",
                    "to": "now"
                },
                "refresh": "5m"
            },
            "folderId": 0,
            "overwrite": True
        }
        
        # Save dashboard configuration
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = output_path / f"grafana_dashboard_{timestamp}.json"
        
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        self.logger.info(f"Grafana dashboard configuration exported to: {dashboard_file}")
        return str(dashboard_file)

def main():
    """Main function for metrics export."""
    parser = argparse.ArgumentParser(description="Export quality metrics for Grafana")
    parser.add_argument('--input', required=True, help='Input quality report JSON file')
    parser.add_argument('--out', default='quality/metrics', help='Output directory')
    parser.add_argument('--formats', nargs='+', choices=['prometheus', 'csv', 'all'],
                       default=['all'], help='Export formats')
    parser.add_argument('--grafana-dashboard', action='store_true',
                       help='Generate Grafana dashboard JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize exporter
    exporter = QualityMetricsExporter()
    
    try:
        # Determine formats
        if 'all' in args.formats:
            formats = ['prometheus', 'csv']
        else:
            formats = args.formats
        
        # Export metrics
        exported_files = exporter.export_metrics(
            report_path=args.input,
            output_dir=args.out,
            formats=formats
        )
        
        print(f"\n✅ Quality metrics exported successfully!")
        for format_type, file_path in exported_files.items():
            if format_type != 'summary':
                print(f"{format_type.upper()}: {file_path}")
        
        # Generate Grafana dashboard if requested
        if args.grafana_dashboard:
            dashboard_file = exporter.generate_grafana_dashboard_json(args.out)
            print(f"Grafana Dashboard: {dashboard_file}")
        
        print(f"\n📊 Metrics ready for Grafana import")
        print(f"Use the CSV or Prometheus files to configure your data sources")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error exporting metrics: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
