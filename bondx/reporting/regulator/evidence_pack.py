#!/usr/bin/env python3
"""
Regulator Mode Evidence Pack Generator

Generates human-readable PDF evidence packs when "regulator mode" is enabled,
bundling validation outcomes, breaches, sector summaries, ESG gaps, and appendices
with data provenance.

Usage:
    python -m bondx.reporting.regulator.evidence_pack --input quality/last_run_report.json --out reports/regulator/ --mode strict
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

class EvidencePackGenerator:
    """Generates regulator-ready evidence packs from quality reports."""
    
    def __init__(self, mode: str = "strict"):
        """Initialize the evidence pack generator."""
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Mode-specific configurations
        self.mode_configs = {
            "strict": {
                "include_all_details": True,
                "highlight_breaches": True,
                "include_recommendations": True,
                "audit_trail": True
            },
            "exploratory": {
                "include_all_details": False,
                "highlight_breaches": True,
                "include_recommendations": False,
                "audit_trail": False
            }
        }
        
        self.config = self.mode_configs.get(mode, self.mode_configs["strict"])
    
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
    
    def load_metrics(self, metrics_path: str) -> Dict[str, Any]:
        """Load quality metrics from JSON file."""
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            self.logger.info(f"Loaded quality metrics: {metrics_path}")
            return metrics
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Could not load metrics {metrics_path}: {e}")
            return {}
    
    def generate_executive_summary(self, report: Dict[str, Any], 
                                  metrics: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        # Extract key statistics
        validation_results = report.get('validation_results', [])
        gate_results = report.get('gate_results', {})
        
        # Count by severity
        severity_counts = {}
        for result in validation_results:
            severity = result.get('severity', 'UNKNOWN')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        total_validations = len(validation_results)
        pass_count = severity_counts.get('PASS', 0)
        warn_count = severity_counts.get('WARN', 0)
        fail_count = severity_counts.get('FAIL', 0)
        
        # Count gate results
        gate_pass = sum(1 for g in gate_results.values() if g.get('status') == 'PASS')
        gate_warn = sum(1 for g in gate_results.values() if g.get('status') == 'WARN')
        gate_fail = sum(1 for g in gate_results.values() if g.get('status') == 'FAIL')
        total_gates = len(gate_results)
        
        # Determine overall status
        if fail_count > 0:
            overall_status = "FAIL"
            status_description = "Quality standards not met"
        elif warn_count > 0:
            overall_status = "WARN"
            status_description = "Quality standards met with warnings"
        else:
            overall_status = "PASS"
            status_description = "Quality standards fully met"
        
        # Top breaches
        top_breaches = []
        for result in validation_results:
            if result.get('severity') in ['FAIL', 'WARN']:
                breach = {
                    'rule': result.get('rule_name', 'Unknown'),
                    'severity': result.get('severity', 'Unknown'),
                    'count': result.get('row_count', 0),
                    'message': result.get('message', 'No details')
                }
                top_breaches.append(breach)
        
        # Sort by severity and count
        top_breaches.sort(key=lambda x: (x['severity'] == 'FAIL', x['count']), reverse=True)
        top_breaches = top_breaches[:5]  # Top 5
        
        # Sectors impacted
        sectors_impacted = set()
        for result in validation_results:
            if result.get('severity') in ['FAIL', 'WARN']:
                dataset = result.get('dataset', '')
                if 'sector' in dataset.lower():
                    sectors_impacted.add(dataset)
        
        summary = f"""EXECUTIVE SUMMARY

OVERALL STATUS: {overall_status}
Description: {status_description}

QUALITY METRICS:
- Total Validations: {total_validations:,}
- PASS: {pass_count:,} ({pass_count/total_validations*100:.1f}%)
- WARN: {warn_count:,} ({warn_count/total_validations*100:.1f}%)
- FAIL: {fail_count:,} ({fail_count/total_validations*100:.1f}%)

QUALITY GATES:
- Total Gates: {total_gates}
- PASS: {gate_pass} ({gate_pass/total_gates*100:.1f}%)
- WARN: {gate_warn} ({gate_warn/total_gates*100:.1f}%)
- FAIL: {gate_fail} ({gate_fail/total_gates*100:.1f}%)

KEY PERFORMANCE INDICATORS:
- Data Coverage: {metrics.get('coverage_percent', 'N/A')}%
- ESG Completeness: {metrics.get('esg_completeness_percent', 'N/A')}%
- Liquidity Index Median: {metrics.get('liquidity_index_median', 'N/A')}
- Data Freshness: {metrics.get('data_freshness_minutes', 'N/A')} minutes

TOP BREACHES:
"""
        
        for i, breach in enumerate(top_breaches, 1):
            summary += f"{i}. {breach['rule']} ({breach['severity']}) - {breach['count']:,} records\n"
            summary += f"   {breach['message']}\n"
        
        if sectors_impacted:
            summary += f"\nSECTORS IMPACTED:\n"
            for sector in sorted(sectors_impacted):
                summary += f"- {sector}\n"
        
        summary += f"\nGENERATION TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return summary
    
    def generate_gate_outcomes_table(self, gate_results: Dict[str, Any]) -> str:
        """Generate quality gate outcomes table."""
        if not gate_results:
            return "No quality gate results available."
        
        # Group gates by status
        gates_by_status = {'PASS': [], 'WARN': [], 'FAIL': []}
        
        for gate_name, gate_result in gate_results.items():
            status = gate_result.get('status', 'UNKNOWN')
            if status in gates_by_status:
                gates_by_status[status].append({
                    'name': gate_name,
                    'details': gate_result
                })
        
        table = "QUALITY GATE OUTCOMES\n"
        table += "=" * 80 + "\n\n"
        
        for status in ['FAIL', 'WARN', 'PASS']:
            gates = gates_by_status[status]
            if gates:
                table += f"{status} GATES ({len(gates)}):\n"
                table += "-" * 40 + "\n"
                
                for gate in gates:
                    table += f"Gate: {gate['name']}\n"
                    details = gate['details']
                    
                    if 'threshold' in details:
                        table += f"Threshold: {details['threshold']}\n"
                    if 'actual' in details:
                        table += f"Actual: {details['actual']}\n"
                    if 'message' in details:
                        table += f"Message: {details['message']}\n"
                    
                    table += "\n"
        
        return table
    
    def generate_esg_section(self, report: Dict[str, Any], 
                            metrics: Dict[str, Any]) -> str:
        """Generate ESG section with completeness and gaps."""
        esg_section = "ENVIRONMENTAL, SOCIAL & GOVERNANCE (ESG) ANALYSIS\n"
        esg_section += "=" * 80 + "\n\n"
        
        # ESG completeness by sector
        esg_completeness = metrics.get('esg_completeness_percent', 0)
        esg_section += f"OVERALL ESG COMPLETENESS: {esg_completeness:.1f}%\n\n"
        
        # ESG validation results
        esg_validations = []
        for result in report.get('validation_results', []):
            if 'esg' in result.get('rule_name', '').lower():
                esg_validations.append(result)
        
        if esg_validations:
            esg_section += "ESG VALIDATION RESULTS:\n"
            esg_section += "-" * 40 + "\n"
            
            for validation in esg_validations:
                esg_section += f"Rule: {validation.get('rule_name', 'Unknown')}\n"
                esg_section += f"Status: {validation.get('severity', 'Unknown')}\n"
                esg_section += f"Records: {validation.get('row_count', 0):,}\n"
                esg_section += f"Message: {validation.get('message', 'No details')}\n\n"
        
        # ESG gaps analysis
        if esg_completeness < 100:
            esg_section += "ESG DATA GAPS IDENTIFIED:\n"
            esg_section += "-" * 40 + "\n"
            
            if esg_completeness < 80:
                esg_section += "⚠️  CRITICAL: ESG completeness below 80% threshold\n"
                esg_section += "   Risk of greenwashing flags in regulatory reporting\n"
                esg_section += "   Immediate action required to improve data collection\n\n"
            elif esg_completeness < 90:
                esg_section += "⚠️  WARNING: ESG completeness below 90% threshold\n"
                esg_section += "   May impact regulatory compliance scores\n"
                esg_section += "   Review data collection processes\n\n"
            else:
                esg_section += "ℹ️  INFO: ESG completeness above 90% but below 100%\n"
                esg_section += "   Minor gaps identified, monitor for improvement\n\n"
        
        # ESG recommendations
        if self.config['include_recommendations']:
            esg_section += "ESG IMPROVEMENT RECOMMENDATIONS:\n"
            esg_section += "-" * 40 + "\n"
            
            if esg_completeness < 80:
                esg_section += "1. Implement mandatory ESG data collection for all issuers\n"
                esg_section += "2. Establish data quality monitoring for ESG metrics\n"
                esg_section += "3. Develop ESG data validation rules and alerts\n"
                esg_section += "4. Consider third-party ESG data providers for gaps\n"
            elif esg_completeness < 90:
                esg_section += "1. Review ESG data collection processes\n"
                esg_section += "2. Implement automated ESG data validation\n"
                esg_section += "3. Establish ESG data quality KPIs\n"
            else:
                esg_section += "1. Continue monitoring ESG data quality\n"
                esg_section += "2. Identify opportunities for ESG data enhancement\n"
        
        return esg_section
    
    def generate_liquidity_section(self, report: Dict[str, Any], 
                                  metrics: Dict[str, Any]) -> str:
        """Generate liquidity analysis section."""
        liquidity_section = "LIQUIDITY RISK ANALYSIS\n"
        liquidity_section += "=" * 80 + "\n\n"
        
        # Liquidity metrics
        liquidity_median = metrics.get('liquidity_index_median', 0)
        liquidity_section += f"LIQUIDITY INDEX MEDIAN: {liquidity_median:.1f}\n\n"
        
        # Liquidity validation results
        liquidity_validations = []
        for result in report.get('validation_results', []):
            if 'liquidity' in result.get('rule_name', '').lower():
                liquidity_validations.append(result)
        
        if liquidity_validations:
            liquidity_section += "LIQUIDITY VALIDATION RESULTS:\n"
            liquidity_section += "-" * 40 + "\n"
            
            for validation in liquidity_validations:
                liquidity_section += f"Rule: {validation.get('rule_name', 'Unknown')}\n"
                liquidity_section += f"Status: {validation.get('severity', 'Unknown')}\n"
                liquidity_section += f"Records: {validation.get('row_count', 0):,}\n"
                liquidity_section += f"Message: {validation.get('message', 'No details')}\n\n"
        
        # Liquidity risk assessment
        if liquidity_median < 30:
            liquidity_section += "🚨 HIGH LIQUIDITY RISK:\n"
            liquidity_section += "   Median liquidity index below 30 threshold\n"
            liquidity_section += "   Significant risk of trading difficulties\n"
            liquidity_section += "   Immediate risk management review required\n\n"
        elif liquidity_median < 50:
            liquidity_section += "⚠️  MODERATE LIQUIDITY RISK:\n"
            liquidity_section += "   Median liquidity index below 50 threshold\n"
            liquidity_section += "   Some risk of trading difficulties\n"
            liquidity_section += "   Monitor and consider risk mitigation\n\n"
        else:
            liquidity_section += "✅ LOW LIQUIDITY RISK:\n"
            liquidity_section += "   Median liquidity index above 50 threshold\n"
            liquidity_section += "   Generally good market liquidity\n\n"
        
        # Low liquidity clusters
        low_liquidity_count = 0
        for result in report.get('validation_results', []):
            if 'liquidity' in result.get('rule_name', '').lower() and result.get('severity') in ['WARN', 'FAIL']:
                low_liquidity_count += result.get('row_count', 0)
        
        if low_liquidity_count > 0:
            liquidity_section += f"LOW LIQUIDITY CLUSTERS:\n"
            liquidity_section += "-" * 40 + "\n"
            liquidity_section += f"Total records with liquidity concerns: {low_liquidity_count:,}\n\n"
        
        # TTE (Time to Exit) risk
        if liquidity_median < 40:
            liquidity_section += "TIME TO EXIT (TTE) RISK:\n"
            liquidity_section += "-" * 40 + "\n"
            liquidity_section += "Low liquidity may impact ability to exit positions quickly\n"
            liquidity_section += "Consider implementing position size limits\n"
            liquidity_section += "Review risk management policies for illiquid positions\n\n"
        
        # Mitigation suggestions
        if self.config['include_recommendations']:
            liquidity_section += "LIQUIDITY RISK MITIGATION SUGGESTIONS:\n"
            liquidity_section += "-" * 40 + "\n"
            
            if liquidity_median < 30:
                liquidity_section += "1. Implement strict position size limits for low-liquidity instruments\n"
                liquidity_section += "2. Establish maximum holding periods for illiquid positions\n"
                liquidity_section += "3. Develop emergency liquidity protocols\n"
                liquidity_section += "4. Consider portfolio-level liquidity stress testing\n"
            elif liquidity_median < 50:
                liquidity_section += "1. Monitor liquidity metrics more frequently\n"
                liquidity_section += "2. Implement gradual position reduction for low-liquidity holdings\n"
                liquidity_section += "3. Review and update liquidity risk policies\n"
            else:
                liquidity_section += "1. Continue monitoring liquidity trends\n"
                liquidity_section += "2. Maintain current risk management practices\n"
        
        return liquidity_section
    
    def generate_appendices(self, report: Dict[str, Any], 
                           metrics: Dict[str, Any]) -> str:
        """Generate appendices with detailed information."""
        appendices = "APPENDICES\n"
        appendices += "=" * 80 + "\n\n"
        
        # Appendix A: Validator List
        appendices += "APPENDIX A: VALIDATION RULES AND RESULTS\n"
        appendices += "-" * 40 + "\n\n"
        
        validation_results = report.get('validation_results', [])
        if validation_results:
            # Group by rule
            rules_summary = {}
            for result in validation_results:
                rule_name = result.get('rule_name', 'Unknown')
                if rule_name not in rules_summary:
                    rules_summary[rule_name] = {
                        'total': 0,
                        'pass': 0,
                        'warn': 0,
                        'fail': 0,
                        'examples': []
                    }
                
                rules_summary[rule_name]['total'] += 1
                severity = result.get('severity', 'UNKNOWN')
                if severity in rules_summary[rule_name]:
                    rules_summary[rule_name][severity.lower()] += 1
                
                # Store example violations
                if severity in ['WARN', 'FAIL'] and len(rules_summary[rule_name]['examples']) < 3:
                    rules_summary[rule_name]['examples'].append({
                        'severity': severity,
                        'count': result.get('row_count', 0),
                        'message': result.get('message', 'No details')
                    })
            
            for rule_name, summary in rules_summary.items():
                appendices += f"Rule: {rule_name}\n"
                appendices += f"Total Validations: {summary['total']}\n"
                appendices += f"PASS: {summary['pass']}, WARN: {summary['warn']}, FAIL: {summary['fail']}\n"
                
                if summary['examples']:
                    appendices += "Example Violations:\n"
                    for example in summary['examples']:
                        appendices += f"  - {example['severity']}: {example['count']} records - {example['message']}\n"
                
                appendices += "\n"
        
        # Appendix B: Policy Version and References
        appendices += "APPENDIX B: POLICY VERSION AND REFERENCES\n"
        appendices += "-" * 40 + "\n\n"
        
        try:
            policy_version_file = Path("bondx/quality/policy_version.txt")
            if policy_version_file.exists():
                with open(policy_version_file, 'r') as f:
                    policy_version = f.read().strip()
                appendices += f"Quality Policy Version: {policy_version}\n"
            else:
                appendices += "Quality Policy Version: Not available\n"
        except Exception as e:
            appendices += f"Quality Policy Version: Error reading - {e}\n"
        
        appendices += f"Evidence Pack Generated: {datetime.now().isoformat()}\n"
        appendices += f"Generation Mode: {self.mode.upper()}\n\n"
        
        # Appendix C: Data Lineage
        appendices += "APPENDIX C: DATA LINEAGE AND PROVENANCE\n"
        appendices += "-" * 40 + "\n\n"
        
        appendices += "Data Sources:\n"
        appendices += "- Quality validation results from BondX quality pipeline\n"
        appendices += "- Metrics aggregated from validation outcomes\n"
        appendices += "- Policy configuration from quality policy files\n\n"
        
        appendices += "Data Processing:\n"
        appendices += "- Validation results processed by BondX validators\n"
        appendices += "- Quality gates evaluated against configured thresholds\n"
        appendices += "- Metrics calculated from validation outcomes\n"
        appendices += "- Evidence pack generated by automated reporting system\n\n"
        
        # Appendix D: Run Timestamps
        appendices += "APPENDIX D: RUN TIMESTAMPS AND METADATA\n"
        appendices += "-" * 40 + "\n\n"
        
        if 'timestamp' in report:
            appendices += f"Quality Run Timestamp: {report['timestamp']}\n"
        if 'generated_at' in report:
            appendices += f"Report Generated: {report['generated_at']}\n"
        
        appendices += f"Evidence Pack Generated: {datetime.now().isoformat()}\n"
        appendices += f"Total Records Processed: {len(validation_results):,}\n"
        appendices += f"Quality Gates Evaluated: {len(report.get('gate_results', {}))}\n"
        
        return appendices
    
    def generate_html_report(self, report: Dict[str, Any], 
                            metrics: Dict[str, Any]) -> str:
        """Generate HTML version of the evidence pack."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BondX Quality Evidence Pack - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 10px; }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-warn {{ color: #f39c12; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .table th {{ background-color: #f2f2f2; }}
        .highlight {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .critical {{ background-color: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BondX Quality Assurance Evidence Pack</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Mode: {self.mode.upper()}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <pre>{self.generate_executive_summary(report, metrics)}</pre>
    </div>
    
    <div class="section">
        <h2>Quality Gate Outcomes</h2>
        <pre>{self.generate_gate_outcomes_table(report.get('gate_results', {}))}</pre>
    </div>
    
    <div class="section">
        <h2>ESG Analysis</h2>
        <pre>{self.generate_esg_section(report, metrics)}</pre>
    </div>
    
    <div class="section">
        <h2>Liquidity Risk Analysis</h2>
        <pre>{self.generate_liquidity_section(report, metrics)}</pre>
    </div>
    
    <div class="section">
        <h2>Appendices</h2>
        <pre>{self.generate_appendices(report, metrics)}</pre>
    </div>
</body>
</html>"""
        
        return html
    
    def generate_evidence_pack(self, report_path: str, metrics_path: Optional[str] = None,
                              output_dir: str = "reports/regulator") -> Dict[str, str]:
        """Generate complete evidence pack."""
        # Load data
        report = self.load_quality_report(report_path)
        metrics = self.load_metrics(metrics_path) if metrics_path else {}
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate sections
        executive_summary = self.generate_executive_summary(report, metrics)
        gate_outcomes = self.generate_gate_outcomes_table(report.get('gate_results', {}))
        esg_section = self.generate_esg_section(report, metrics)
        liquidity_section = self.generate_liquidity_section(report, metrics)
        appendices = self.generate_appendices(report, metrics)
        
        # Combine into full report
        full_report = f"""BONDX QUALITY ASSURANCE EVIDENCE PACK
{'=' * 80}

{executive_summary}

{'=' * 80}

{gate_outcomes}

{'=' * 80}

{esg_section}

{'=' * 80}

{liquidity_section}

{'=' * 80}

{appendices}
"""
        
        # Save text version
        text_file = output_path / f"evidence_pack_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        # Save HTML version
        html_content = self.generate_html_report(report, metrics)
        html_file = output_path / f"evidence_pack_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate PDF (stub for future implementation)
        pdf_file = output_path / f"evidence_pack_{timestamp}.pdf"
        try:
            # This would use wkhtmltopdf or similar in production
            self.logger.info(f"PDF generation not implemented yet. HTML version available: {html_file}")
        except Exception as e:
            self.logger.warning(f"PDF generation failed: {e}")
        
        # Save metadata
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'input_report': report_path,
            'input_metrics': metrics_path,
            'output_files': {
                'text': str(text_file),
                'html': str(html_file),
                'pdf': str(pdf_file)
            },
            'config': self.config
        }
        
        metadata_file = output_path / f"evidence_pack_{timestamp}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Evidence pack generated successfully")
        self.logger.info(f"Text version: {text_file}")
        self.logger.info(f"HTML version: {html_file}")
        self.logger.info(f"Metadata: {metadata_file}")
        
        return {
            'text': str(text_file),
            'html': str(html_file),
            'pdf': str(pdf_file),
            'metadata': str(metadata_file)
        }

def main():
    """Main function for evidence pack generation."""
    parser = argparse.ArgumentParser(description="Generate regulator evidence pack")
    parser.add_argument('--input', required=True, help='Input quality report JSON file')
    parser.add_argument('--metrics', help='Input metrics JSON file (optional)')
    parser.add_argument('--out', default='reports/regulator', help='Output directory')
    parser.add_argument('--mode', choices=['strict', 'exploratory'], default='strict',
                       help='Generation mode (strict=full details, exploratory=summary)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize generator
    generator = EvidencePackGenerator(args.mode)
    
    try:
        # Generate evidence pack
        output_files = generator.generate_evidence_pack(
            report_path=args.input,
            metrics_path=args.metrics,
            output_dir=args.out
        )
        
        print(f"\n✅ Evidence pack generated successfully!")
        print(f"Text version: {output_files['text']}")
        print(f"HTML version: {output_files['html']}")
        print(f"Metadata: {output_files['metadata']}")
        
        if args.mode == 'strict':
            print("\n📋 Strict mode: Full details included for regulatory compliance")
        else:
            print("\n🔍 Exploratory mode: Summary view for analysis")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating evidence pack: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
