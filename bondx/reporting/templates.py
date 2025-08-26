"""
Template Manager

This module manages report templates including:
- Template loading and customization
- Branding and styling options
- Template validation and testing
- Dynamic template generation
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from jinja2 import Environment, FileSystemLoader, Template, TemplateError
import yaml

logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages report templates and customization"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize template manager"""
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Available template types
        self.template_types = {
            'investor': 'investor_report.html',
            'regulator': 'regulator_report.html',
            'executive': 'executive_summary.html',
            'technical': 'technical_analysis.html',
            'compliance': 'compliance_report.html'
        }
        
        # Default branding
        self.default_branding = {
            'company_name': 'BondX Analytics',
            'primary_color': '#1f77b4',
            'secondary_color': '#ff7f0e',
            'logo_path': None,
            'font_family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
            'header_style': 'gradient'
        }
        
        # Load custom branding if available
        self.branding = self._load_branding()
        
        # Initialize templates
        self._ensure_templates_exist()
    
    def _load_branding(self) -> Dict[str, Any]:
        """Load custom branding configuration"""
        branding_file = self.template_dir / "branding.yaml"
        
        if branding_file.exists():
            try:
                with open(branding_file, 'r') as f:
                    custom_branding = yaml.safe_load(f)
                    # Merge with defaults
                    branding = self.default_branding.copy()
                    branding.update(custom_branding)
                    return branding
            except Exception as e:
                logger.warning(f"Failed to load branding config: {e}")
                return self.default_branding
        else:
            return self.default_branding
    
    def _ensure_templates_exist(self):
        """Ensure all required templates exist"""
        for template_type, filename in self.template_types.items():
            template_path = self.template_dir / filename
            if not template_path.exists():
                self._create_default_template(template_type, filename)
    
    def _create_default_template(self, template_type: str, filename: str):
        """Create default template for a given type"""
        if template_type == 'investor':
            self._create_investor_template(filename)
        elif template_type == 'regulator':
            self._create_regulator_template(filename)
        elif template_type == 'executive':
            self._create_executive_template(filename)
        elif template_type == 'technical':
            self._create_technical_template(filename)
        elif template_type == 'compliance':
            self._create_compliance_template(filename)
    
    def _create_investor_template(self, filename: str):
        """Create default investor template"""
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
        
        <footer class="report-footer">
            <p>{{ branding.company_name }} | Generated on {{ report_date }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(self.template_dir / filename, "w") as f:
            f.write(template_content)
    
    def _create_regulator_template(self, filename: str):
        """Create default regulator template"""
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
                    </table>
                </div>
            </div>
        </section>
        
        <footer class="report-footer">
            <p>{{ branding.company_name }} | Regulatory Compliance Report | Generated on {{ report_date }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(self.template_dir / filename, "w") as f:
            f.write(template_content)
    
    def _create_executive_template(self, filename: str):
        """Create default executive summary template"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }} - Executive Summary</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{{ config.title }}</h1>
            <h2>Executive Summary</h2>
            <p class="report-meta">
                Generated: {{ report_date }} | Portfolio: {{ config.portfolio_id or 'All Portfolios' }}
            </p>
        </header>
        
        <section class="executive-summary">
            <h3>Key Highlights</h3>
            <div class="highlights-grid">
                <div class="highlight-card">
                    <h4>Portfolio Performance</h4>
                    <p>Total Value: ${{ "{:,.0f}".format(portfolio_summary.total_value) }}</p>
                    <p>Risk Level: {{ portfolio_summary.risk_level }}</p>
                </div>
            </div>
        </section>
        
        <footer class="report-footer">
            <p>{{ branding.company_name }} | Executive Summary | Generated on {{ report_date }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(self.template_dir / filename, "w") as f:
            f.write(template_content)
    
    def _create_technical_template(self, filename: str):
        """Create default technical analysis template"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }} - Technical Analysis</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{{ config.title }}</h1>
            <h2>Technical Analysis Report</h2>
            <p class="report-meta">
                Generated: {{ report_date }} | Portfolio: {{ config.portfolio_id or 'All Portfolios' }}
            </p>
        </header>
        
        <section class="technical-analysis">
            <h3>Technical Indicators</h3>
            <div class="indicators-grid">
                <div class="indicator-card">
                    <h4>Liquidity Metrics</h4>
                    <p>Average Liquidity: {{ "%.2f"|format(portfolio_summary.avg_liquidity) }}</p>
                    <p>Liquidity Coverage: {{ "%.1f"|format(portfolio_summary.liquidity_coverage) }}%</p>
                </div>
            </div>
        </section>
        
        <footer class="report-footer">
            <p>{{ branding.company_name }} | Technical Analysis | Generated on {{ report_date }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(self.template_dir / filename, "w") as f:
            f.write(template_content)
    
    def _create_compliance_template(self, filename: str):
        """Create default compliance report template"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }} - Compliance Report</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{{ config.title }}</h1>
            <h2>Compliance Report</h2>
            <p class="report-meta">
                Generated: {{ report_date }} | Portfolio: {{ config.portfolio_id or 'All Portfolios' }}
            </p>
        </header>
        
        <section class="compliance-status">
            <h3>Compliance Status</h3>
            <div class="status-grid">
                <div class="status-card">
                    <h4>Regulatory Requirements</h4>
                    <p>Status: {{ portfolio_summary.compliance_status }}</p>
                    <p>Last Review: {{ portfolio_summary.last_compliance_review }}</p>
                </div>
            </div>
        </section>
        
        <footer class="report-footer">
            <p>{{ branding.company_name }} | Compliance Report | Generated on {{ report_date }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(self.template_dir / filename, "w") as f:
            f.write(template_content)
    
    def get_template(self, template_type: str) -> Optional[Template]:
        """Get a template by type"""
        if template_type not in self.template_types:
            logger.error(f"Unknown template type: {template_type}")
            return None
        
        filename = self.template_types[template_type]
        template_path = self.template_dir / filename
        
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            return None
        
        try:
            return self.env.get_template(filename)
        except TemplateError as e:
            logger.error(f"Failed to load template {filename}: {e}")
            return None
    
    def render_template(self, template_type: str, data: Dict[str, Any]) -> Optional[str]:
        """Render a template with data"""
        template = self.get_template(template_type)
        if not template:
            return None
        
        try:
            # Add branding to data
            render_data = data.copy()
            render_data['branding'] = self.branding
            
            return template.render(**render_data)
        except TemplateError as e:
            logger.error(f"Failed to render template {template_type}: {e}")
            return None
    
    def update_branding(self, new_branding: Dict[str, Any]):
        """Update branding configuration"""
        self.branding.update(new_branding)
        
        # Save to file
        branding_file = self.template_dir / "branding.yaml"
        try:
            with open(branding_file, 'w') as f:
                yaml.dump(self.branding, f, default_flow_style=False)
            logger.info("Branding configuration updated and saved")
        except Exception as e:
            logger.error(f"Failed to save branding configuration: {e}")
    
    def validate_template(self, template_type: str) -> bool:
        """Validate a template by attempting to render it with sample data"""
        sample_data = {
            'config': {
                'title': 'Test Report',
                'subtitle': 'Test Subtitle',
                'portfolio_id': 'TEST001'
            },
            'report_date': '2024-01-01',
            'portfolio_summary': {
                'total_bonds': 100,
                'total_value': 1000000,
                'avg_liquidity': 0.75,
                'high_risk_count': 5,
                'anomaly_count': 2,
                'stress_impact': 2.5
            },
            'top_actions': [
                {
                    'title': 'Test Action',
                    'description': 'Test description',
                    'priority': 'medium',
                    'urgency': 'Within 1 week'
                }
            ]
        }
        
        try:
            rendered = self.render_template(template_type, sample_data)
            return rendered is not None and len(rendered) > 0
        except Exception as e:
            logger.error(f"Template validation failed for {template_type}: {e}")
            return False
    
    def list_templates(self) -> List[str]:
        """List available template types"""
        return list(self.template_types.keys())
    
    def get_template_info(self, template_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a template"""
        if template_type not in self.template_types:
            return None
        
        filename = self.template_types[template_type]
        template_path = self.template_dir / filename
        
        if not template_path.exists():
            return None
        
        try:
            stat = template_path.stat()
            return {
                'type': template_type,
                'filename': filename,
                'path': str(template_path),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'exists': True
            }
        except Exception as e:
            logger.error(f"Failed to get template info for {template_type}: {e}")
            return None
    
    def create_custom_template(self, template_type: str, content: str) -> bool:
        """Create a custom template"""
        if template_type in self.template_types:
            logger.warning(f"Template type {template_type} already exists, overwriting")
        
        try:
            # Validate template syntax
            Template(content)
            
            # Save template
            filename = f"{template_type}_report.html"
            template_path = self.template_dir / filename
            
            with open(template_path, 'w') as f:
                f.write(content)
            
            # Update template types
            self.template_types[template_type] = filename
            
            logger.info(f"Custom template created: {template_type}")
            return True
            
        except TemplateError as e:
            logger.error(f"Invalid template syntax: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to create custom template: {e}")
            return False
    
    def backup_templates(self, backup_dir: str) -> bool:
        """Backup all templates to a directory"""
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            for template_type, filename in self.template_types.items():
                source_path = self.template_dir / filename
                if source_path.exists():
                    backup_file = backup_path / f"{template_type}_{filename}"
                    with open(source_path, 'r') as src, open(backup_file, 'w') as dst:
                        dst.write(src.read())
            
            # Backup branding
            branding_file = self.template_dir / "branding.yaml"
            if branding_file.exists():
                backup_branding = backup_path / "branding.yaml"
                with open(branding_file, 'r') as src, open(backup_branding, 'w') as dst:
                    dst.write(src.read())
            
            logger.info(f"Templates backed up to {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Template backup failed: {e}")
            return False
    
    def restore_templates(self, backup_dir: str) -> bool:
        """Restore templates from backup"""
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            # Restore branding first
            backup_branding = backup_path / "branding.yaml"
            if backup_branding.exists():
                target_branding = self.template_dir / "branding.yaml"
                with open(backup_branding, 'r') as src, open(target_branding, 'w') as dst:
                    dst.write(src.read())
                
                # Reload branding
                self.branding = self._load_branding()
            
            # Restore templates
            for template_type, filename in self.template_types.items():
                backup_file = backup_path / f"{template_type}_{filename}"
                if backup_file.exists():
                    target_file = self.template_dir / filename
                    with open(backup_file, 'r') as src, open(target_file, 'w') as dst:
                        dst.write(src.read())
            
            logger.info(f"Templates restored from {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Template restore failed: {e}")
            return False
