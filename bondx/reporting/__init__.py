"""
Reporting Module

This module provides automated reporting capabilities for BondX including:
- HTML report generation
- PDF export
- Investor and regulator views
- Dashboard-ready JSON exports
"""

from .generate_reports import ReportGenerator
from .templates import TemplateManager

__all__ = [
    'ReportGenerator',
    'TemplateManager'
]
