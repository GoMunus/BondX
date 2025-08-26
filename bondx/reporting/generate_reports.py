"""
Report Generator

This module generates automated Liquidity Analysis Reports including:
- Investor view: concise insights and top actions
- Regulator view: detailed analysis and compliance metrics
- HTML and PDF export capabilities
- Configurable branding and styling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from jinja2 import Template, Environment, FileSystemLoader
import webbrowser
import tempfile

# For PDF generation
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str = "Liquidity Analysis Report"
    subtitle: str = "BondX Portfolio Analysis"
    company_name: str = "BondX Analytics"
    logo_path: Optional[str] = None
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    include_charts: bool = True
    include_tables: bool = True
    max_issuers_shown: int = 20
    report_date: Optional[datetime] = None
    portfolio_id: Optional[str] = None

@dataclass
class ReportData:
    """Data structure for report generation"""
    liquidity_insights: Dict[str, Any]
    stress_test_results: Dict[str, Any]
    heatmap_data: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    anomaly_results: List[Any]
    portfolio_summary: Dict[str, Any]

class ReportGenerator:
    """Generates comprehensive liquidity analysis reports"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the report generator"""
        self.config = config or ReportConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default HTML templates if they don't exist"""
        # Investor template
        investor_template = self.template_dir / "investor_report.html"
        if not investor_template.exists():
            self._create_investor_template()
        
        # Regulator template
        regulator_template = self.template_dir / "regulator_report.html"
        if not regulator_template.exists():
            self._create_regulator_template()
        
        # CSS template
        css_template = self.template_dir / "styles.css"
        if not css_template.exists():
            self._create_css_template()
    
    def _create_investor_template(self):
        """Create default investor report template"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }} - Investor View</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{{ config.title }}</h1>
            <h2>{{ config.subtitle }}</h2>
            <p class="report-meta">
                Generated: {{ report_date }} | Portfolio: {{ config.portfolio_id or 'All Portfolios' }}
            </p>
        </header>
        
        <section class="executive-summary">
            <h3>Executive Summary</h3>
            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Portfolio Overview</h4>
                    <p>Total Bonds: {{ portfolio_summary.total_bonds }}</p>
                    <p>Total Value: ${{ "{:,.0f}".format(portfolio_summary.total_value) }}</p>
                    <p>Average Liquidity: {{ "%.2f"|format(portfolio_summary.avg_liquidity) }}</p>
                </div>
                <div class="summary-card">
                    <h4>Risk Indicators</h4>
                    <p>High Risk: {{ portfolio_summary.high_risk_count }}</p>
                    <p>Anomalies: {{ portfolio_summary.anomaly_count }}</p>
                    <p>Stress Impact: {{ "%.1f"|format(portfolio_summary.stress_impact) }}%</p>
                </div>
            </div>
        </section>
        
        <section class="top-actions">
            <h3>Top 5 Actions Required</h3>
            <ol class="action-list">
                {% for action in top_actions %}
                <li class="action-item priority-{{ action.priority }}">
                    <strong>{{ action.title }}</strong>
                    <p>{{ action.description }}</p>
                    <span class="action-urgency">{{ action.urgency }}</span>
                </li>
                {% endfor %}
            </ol>
        </section>
        
        <section class="liquidity-insights">
            <h3>Liquidity Insights</h3>
            <div class="insights-grid">
                <div class="insight-card">
                    <h4>Top Liquid Issuers</h4>
                    <ul>
                        {% for issuer in liquidity_insights.top_liquid_issuers[:5] %}
                        <li>{{ issuer.issuer_name }} - {{ "%.2f"|format(issuer.liquidity_index) }}</li>
                        {% endfor %}
                    </ul>
                </div>
                <div class="insight-card">
                    <h4>Exit Route Recommendations</h4>
                    <ul>
                        {% for route in liquidity_insights.exit_routes[:5] %}
                        <li>{{ route.issuer_name }}: {{ route.recommended_route }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </section>
        
        <section class="risk-alerts">
            <h3>Risk Alerts</h3>
            {% if anomaly_results %}
            <div class="alert-list">
                {% for anomaly in anomaly_results[:5] %}
                <div class="alert-item severity-{{ anomaly.severity.value }}">
                    <h4>{{ anomaly.issuer_name }}</h4>
                    <p>Sector: {{ anomaly.sector }} | Rating: {{ anomaly.credit_rating }}</p>
                    <p>Score: {{ "%.2f"|format(anomaly.anomaly_score) }} | Confidence: {{ "%.1f"|format(anomaly.confidence * 100) }}%</p>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p>No significant anomalies detected.</p>
            {% endif %}
        </section>
        
        <footer class="report-footer">
            <p>{{ config.company_name }} | Generated on {{ report_date }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(self.template_dir / "investor_report.html", "w") as f:
            f.write(template_content)
    
    def _create_regulator_template(self):
        """Create default regulator report template"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }} - Regulator View</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{{ config.title }}</h1>
            <h2>{{ config.subtitle }} - Regulatory Compliance Report</h2>
            <p class="report-meta">
                Generated: {{ report_date }} | Portfolio: {{ config.portfolio_id or 'All Portfolios' }}
            </p>
        </header>
        
        <section class="compliance-overview">
            <h3>Compliance Overview</h3>
            <div class="compliance-grid">
                <div class="compliance-card">
                    <h4>Portfolio Metrics</h4>
                    <table class="data-table">
                        <tr><td>Total Bonds</td><td>{{ portfolio_summary.total_bonds }}</td></tr>
                        <tr><td>Total Value</td><td>${{ "{:,.0f}".format(portfolio_summary.total_value) }}</td></tr>
                        <tr><td>Average Liquidity</td><td>{{ "%.2f"|format(portfolio_summary.avg_liquidity) }}</td></tr>
                        <tr><td>Risk-Weighted Assets</td><td>${{ "{:,.0f}".format(portfolio_summary.risk_weighted_assets) }}</td></tr>
                    </table>
                </div>
                <div class="compliance-card">
                    <h4>Risk Metrics</h4>
                    <table class="data-table">
                        <tr><td>VaR (95%)</td><td>${{ "{:,.0f}".format(portfolio_summary.var_95) }}</td></tr>
                        <tr><td>Expected Shortfall</td><td>${{ "{:,.0f}".format(portfolio_summary.expected_shortfall) }}</td></tr>
                        <tr><td>Liquidity Coverage</td><td>{{ "%.1f"|format(portfolio_summary.liquidity_coverage) }}%</td></tr>
                        <tr><td>Stress Test Impact</td><td>{{ "%.1f"|format(portfolio_summary.stress_impact) }}%</td></tr>
                    </table>
                </div>
            </div>
        </section>
        
        <section class="detailed-analysis">
            <h3>Detailed Analysis</h3>
            
            <div class="analysis-section">
                <h4>Liquidity Distribution by Sector</h4>
                <div class="sector-analysis">
                    {% for sector in liquidity_insights.sector_analysis %}
                    <div class="sector-item">
                        <span class="sector-name">{{ sector.sector }}</span>
                        <span class="sector-liquidity">{{ "%.2f"|format(sector.avg_liquidity) }}</span>
                        <span class="sector-count">{{ sector.bond_count }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="analysis-section">
                <h4>Credit Rating Distribution</h4>
                <div class="rating-analysis">
                    {% for rating in liquidity_insights.rating_analysis %}
                    <div class="rating-item">
                        <span class="rating-grade">{{ rating.rating }}</span>
                        <span class="rating-count">{{ rating.count }}</span>
                        <span class="rating-liquidity">{{ "%.2f"|format(rating.avg_liquidity) }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </section>
        
        <section class="stress-testing">
            <h3>Stress Testing Results</h3>
            {% if stress_test_results.scenarios %}
            <div class="scenario-results">
                {% for scenario in stress_test_results.scenarios %}
                <div class="scenario-item">
                    <h4>{{ scenario.name }}</h4>
                    <p>{{ scenario.description }}</p>
                    <table class="scenario-table">
                        <tr><td>Portfolio Impact</td><td>{{ "%.2f"|format(scenario.portfolio_impact) }}%</td></tr>
                        <tr><td>Liquidity Impact</td><td>{{ "%.2f"|format(scenario.liquidity_impact) }}%</td></tr>
                        <tr><td>Risk Increase</td><td>{{ "%.2f"|format(scenario.risk_increase) }}%</td></tr>
                    </table>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </section>
        
        <section class="anomaly-detection">
            <h3>Anomaly Detection Results</h3>
            {% if anomaly_results %}
            <div class="anomaly-summary">
                <p>Total Anomalies Detected: {{ anomaly_results|length }}</p>
                <div class="severity-breakdown">
                    {% for severity in ['critical', 'high', 'medium', 'low'] %}
                    <span class="severity-{{ severity }}">
                        {{ severity|title }}: {{ anomaly_results|selectattr('severity.value', 'equalto', severity)|list|length }}
                    </span>
                    {% endfor %}
                </div>
            </div>
            
            <div class="anomaly-details">
                {% for anomaly in anomaly_results %}
                <div class="anomaly-item severity-{{ anomaly.severity.value }}">
                    <h4>{{ anomaly.issuer_name }}</h4>
                    <p>Sector: {{ anomaly.sector }} | Rating: {{ anomaly.credit_rating }}</p>
                    <p>Anomaly Score: {{ "%.2f"|format(anomaly.anomaly_score) }}</p>
                    <p>Confidence: {{ "%.1f"|format(anomaly.confidence * 100) }}%</p>
                    <p>Contributing Factors: {{ anomaly.contributing_factors|join(', ') }}</p>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p>No anomalies detected in this analysis period.</p>
            {% endif %}
        </section>
        
        <footer class="report-footer">
            <p>{{ config.company_name }} | Regulatory Compliance Report | Generated on {{ report_date }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(self.template_dir / "regulator_report.html", "w") as f:
            f.write(template_content)
    
    def _create_css_template(self):
        """Create default CSS styling"""
        css_content = """
/* BondX Report Styling */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background-color: white;
    box-shadow: 0 0 20px rgba(0,0,0,0.1);
}

.report-header {
    background: linear-gradient(135deg, #1f77b4, #ff7f0e);
    color: white;
    padding: 2rem;
    text-align: center;
}

.report-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.report-header h2 {
    font-size: 1.5rem;
    font-weight: 300;
    margin-bottom: 1rem;
}

.report-meta {
    font-size: 0.9rem;
    opacity: 0.9;
}

section {
    padding: 2rem;
    border-bottom: 1px solid #e9ecef;
}

section h3 {
    color: #1f77b4;
    margin-bottom: 1.5rem;
    font-size: 1.8rem;
}

.summary-grid, .insights-grid, .compliance-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.summary-card, .insight-card, .compliance-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
}

.summary-card h4, .insight-card h4, .compliance-card h4 {
    color: #1f77b4;
    margin-bottom: 1rem;
}

.action-list {
    list-style: none;
}

.action-item {
    background: #f8f9fa;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 6px;
    border-left: 4px solid #28a745;
}

.action-item.priority-high {
    border-left-color: #dc3545;
}

.action-item.priority-medium {
    border-left-color: #ffc107;
}

.action-urgency {
    display: inline-block;
    background: #dc3545;
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

.insights-grid ul {
    list-style: none;
}

.insights-grid li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #e9ecef;
}

.alert-list {
    display: grid;
    gap: 1rem;
}

.alert-item {
    padding: 1rem;
    border-radius: 6px;
    border-left: 4px solid;
}

.alert-item.severity-critical {
    background: #f8d7da;
    border-left-color: #dc3545;
}

.alert-item.severity-high {
    background: #fff3cd;
    border-left-color: #ffc107;
}

.alert-item.severity-medium {
    background: #d1ecf1;
    border-left-color: #17a2b8;
}

.alert-item.severity-low {
    background: #d4edda;
    border-left-color: #28a745;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
}

.data-table td {
    padding: 0.5rem;
    border-bottom: 1px solid #e9ecef;
}

.data-table td:first-child {
    font-weight: bold;
    color: #495057;
}

.sector-analysis, .rating-analysis {
    display: grid;
    gap: 0.5rem;
}

.sector-item, .rating-item {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 4px;
}

.scenario-results {
    display: grid;
    gap: 1.5rem;
}

.scenario-item {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
}

.scenario-table {
    width: 100%;
    margin-top: 1rem;
}

.anomaly-summary {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 6px;
    margin-bottom: 1.5rem;
}

.severity-breakdown {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.severity-breakdown span {
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.9rem;
    font-weight: bold;
}

.severity-critical { background: #dc3545; color: white; }
.severity-high { background: #ffc107; color: #212529; }
.severity-medium { background: #17a2b8; color: white; }
.severity-low { background: #28a745; color: white; }

.anomaly-details {
    display: grid;
    gap: 1rem;
}

.anomaly-item {
    padding: 1rem;
    border-radius: 6px;
    border-left: 4px solid;
}

.anomaly-item.severity-critical {
    background: #f8d7da;
    border-left-color: #dc3545;
}

.anomaly-item.severity-high {
    background: #fff3cd;
    border-left-color: #ffc107;
}

.anomaly-item.severity-medium {
    background: #d1ecf1;
    border-left-color: #17a2b8;
}

.anomaly-item.severity-low {
    background: #d4edda;
    border-left-color: #28a745;
}

.report-footer {
    background: #343a40;
    color: white;
    text-align: center;
    padding: 1rem;
    font-size: 0.9rem;
}

/* Responsive design */
@media (max-width: 768px) {
    .summary-grid, .insights-grid, .compliance-grid {
        grid-template-columns: 1fr;
    }
    
    .report-header h1 {
        font-size: 2rem;
    }
    
    section {
        padding: 1rem;
    }
}
        """
        
        with open(self.template_dir / "styles.css", "w") as f:
            f.write(css_content)
    
    def generate_investor_report(self, data: ReportData, output_path: Optional[str] = None) -> str:
        """Generate investor-focused report"""
        self.logger.info("Generating investor report")
        
        # Prepare data for investor view
        report_data = self._prepare_investor_data(data)
        
        # Load template
        template = self.env.get_template("investor_report.html")
        
        # Render template
        html_content = template.render(
            config=self.config,
            report_date=self.config.report_date or datetime.now().strftime("%Y-%m-%d %H:%M"),
            **report_data
        )
        
        # Save or return HTML
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            self.logger.info(f"Investor report saved to {output_path}")
            return str(output_file)
        else:
            return html_content
    
    def generate_regulator_report(self, data: ReportData, output_path: Optional[str] = None) -> str:
        """Generate regulator-focused report"""
        self.logger.info("Generating regulator report")
        
        # Prepare data for regulator view
        report_data = self._prepare_regulator_data(data)
        
        # Load template
        template = self.env.get_template("regulator_report.html")
        
        # Render template
        html_content = template.render(
            config=self.config,
            report_date=self.config.report_date or datetime.now().strftime("%Y-%m-%d %H:%M"),
            **report_data
        )
        
        # Save or return HTML
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            self.logger.info(f"Regulator report saved to {output_path}")
            return str(output_file)
        else:
            return html_content
    
    def _prepare_investor_data(self, data: ReportData) -> Dict[str, Any]:
        """Prepare data for investor view"""
        # Extract key insights
        liquidity_insights = data.liquidity_insights
        portfolio_summary = data.portfolio_summary
        
        # Generate top actions
        top_actions = self._generate_top_actions(data)
        
        # Prepare liquidity insights
        top_liquid_issuers = liquidity_insights.get('issuer_rankings', [])[:5]
        exit_routes = liquidity_insights.get('exit_route_recommendations', [])[:5]
        
        # Prepare sector analysis
        sector_analysis = liquidity_insights.get('sector_analysis', [])
        rating_analysis = liquidity_insights.get('rating_analysis', [])
        
        return {
            'top_actions': top_actions,
            'liquidity_insights': {
                'top_liquid_issuers': top_liquid_issuers,
                'exit_routes': exit_routes,
                'sector_analysis': sector_analysis,
                'rating_analysis': rating_analysis
            },
            'portfolio_summary': portfolio_summary,
            'anomaly_results': data.anomaly_results[:5] if data.anomaly_results else []
        }
    
    def _prepare_regulator_data(self, data: ReportData) -> Dict[str, Any]:
        """Prepare data for regulator view"""
        # Extract detailed data
        liquidity_insights = data.liquidity_insights
        stress_test_results = data.stress_test_results
        portfolio_summary = data.portfolio_summary
        
        # Prepare sector analysis
        sector_analysis = liquidity_insights.get('sector_analysis', [])
        rating_analysis = liquidity_insights.get('rating_analysis', [])
        
        # Prepare stress test results
        scenarios = stress_test_results.get('scenarios', [])
        
        return {
            'liquidity_insights': {
                'sector_analysis': sector_analysis,
                'rating_analysis': rating_analysis
            },
            'stress_test_results': {
                'scenarios': scenarios
            },
            'portfolio_summary': portfolio_summary,
            'anomaly_results': data.anomaly_results if data.anomaly_results else []
        }
    
    def _generate_top_actions(self, data: ReportData) -> List[Dict[str, Any]]:
        """Generate top 5 actions for investors"""
        actions = []
        
        # Action 1: High-risk bonds
        high_risk_count = data.portfolio_summary.get('high_risk_count', 0)
        if high_risk_count > 0:
            actions.append({
                'title': 'Review High-Risk Bonds',
                'description': f'Monitor {high_risk_count} bonds with elevated risk indicators',
                'priority': 'high',
                'urgency': 'Immediate'
            })
        
        # Action 2: Anomalies
        anomaly_count = len(data.anomaly_results) if data.anomaly_results else 0
        if anomaly_count > 0:
            actions.append({
                'title': 'Investigate Anomalies',
                'description': f'Review {anomaly_count} bonds flagged for unusual behavior',
                'priority': 'high',
                'urgency': 'Within 24 hours'
            })
        
        # Action 3: Liquidity concerns
        low_liquidity_count = data.portfolio_summary.get('low_liquidity_count', 0)
        if low_liquidity_count > 0:
            actions.append({
                'title': 'Address Liquidity Concerns',
                'description': f'Develop exit strategies for {low_liquidity_count} illiquid positions',
                'priority': 'medium',
                'urgency': 'Within 48 hours'
            })
        
        # Action 4: Sector concentration
        sector_concentration = data.portfolio_summary.get('sector_concentration', {})
        if sector_concentration:
            max_sector = max(sector_concentration.items(), key=lambda x: x[1])
            if max_sector[1] > 0.3:  # More than 30%
                actions.append({
                    'title': 'Reduce Sector Concentration',
                    'description': f'Consider reducing exposure to {max_sector[0]} sector ({max_sector[1]:.1%})',
                    'priority': 'medium',
                    'urgency': 'Within 1 week'
                })
        
        # Action 5: Stress test impact
        stress_impact = data.portfolio_summary.get('stress_impact', 0)
        if stress_impact > 0.05:  # More than 5%
            actions.append({
                'title': 'Stress Test Review',
                'description': f'Portfolio shows {stress_impact:.1%} impact under stress scenarios',
                'priority': 'medium',
                'urgency': 'Within 1 week'
            })
        
        # Fill remaining slots if needed
        while len(actions) < 5:
            actions.append({
                'title': 'Regular Portfolio Review',
                'description': 'Continue monitoring portfolio metrics and market conditions',
                'priority': 'low',
                'urgency': 'Ongoing'
            })
        
        return actions[:5]
    
    def export_to_pdf(self, html_content: str, output_path: str) -> str:
        """Export HTML report to PDF"""
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint is required for PDF export. Install with: pip install weasyprint")
        
        try:
            # Create PDF from HTML
            HTML(string=html_content).write_pdf(output_path)
            self.logger.info(f"PDF exported to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"PDF export failed: {e}")
            raise
    
    def generate_batch_reports(self, data: ReportData, output_dir: str, 
                             portfolio_id: str, timestamp: Optional[str] = None) -> List[str]:
        """Generate batch reports with timestamped filenames"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Generate investor report
        investor_file = output_path / f"investor_report_{portfolio_id}_{timestamp}.html"
        self.generate_investor_report(data, str(investor_file))
        generated_files.append(str(investor_file))
        
        # Generate regulator report
        regulator_file = output_path / f"regulator_report_{portfolio_id}_{timestamp}.html"
        self.generate_regulator_report(data, str(regulator_file))
        generated_files.append(str(regulator_file))
        
        # Generate PDF versions if possible
        try:
            investor_pdf = output_path / f"investor_report_{portfolio_id}_{timestamp}.pdf"
            investor_html = self.generate_investor_report(data)
            self.export_to_pdf(investor_html, str(investor_pdf))
            generated_files.append(str(investor_pdf))
            
            regulator_pdf = output_path / f"regulator_report_{portfolio_id}_{timestamp}.pdf"
            regulator_html = self.generate_regulator_report(data)
            self.export_to_pdf(regulator_html, str(regulator_pdf))
            generated_files.append(str(regulator_pdf))
        except ImportError:
            self.logger.warning("PDF export not available, skipping PDF generation")
        
        self.logger.info(f"Batch reports generated: {len(generated_files)} files")
        return generated_files
