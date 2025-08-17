"""
Risk Models for BondX Risk Management System.

This module defines all data models and enums used in the risk management system.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from uuid import uuid4


class RiskMetricType(Enum):
    """Types of risk metrics."""
    
    VAR = "VAR"
    CVAR = "CVAR"
    VOLATILITY = "VOLATILITY"
    BETA = "BETA"
    CORRELATION = "CORRELATION"
    DURATION = "DURATION"
    CONVEXITY = "CONVEXITY"
    YIELD = "YIELD"
    CREDIT_SPREAD = "CREDIT_SPREAD"
    LIQUIDITY = "LIQUIDITY"
    CONCENTRATION = "CONCENTRATION"
    LEVERAGE = "LEVERAGE"


class RiskLevel(Enum):
    """Risk levels for alerts and monitoring."""
    
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class LimitType(Enum):
    """Types of risk limits."""
    
    POSITION_LIMIT = "POSITION_LIMIT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    VAR_LIMIT = "VAR_LIMIT"
    LOSS_LIMIT = "LOSS_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    LIQUIDITY_LIMIT = "LIQUIDITY_LIMIT"
    CREDIT_LIMIT = "CREDIT_LIMIT"


class ComplianceStatus(Enum):
    """Compliance status values."""
    
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a portfolio or position."""
    
    portfolio_id: str
    calculation_time: datetime = field(default_factory=datetime.utcnow)
    
    # Value at Risk metrics
    var_95_1d: Optional[float] = None
    var_99_1d: Optional[float] = None
    var_95_10d: Optional[float] = None
    var_99_10d: Optional[float] = None
    cvar_95_1d: Optional[float] = None
    cvar_99_1d: Optional[float] = None
    
    # Volatility and correlation metrics
    portfolio_volatility: Optional[float] = None
    portfolio_beta: Optional[float] = None
    max_correlation: Optional[float] = None
    avg_correlation: Optional[float] = None
    
    # Fixed income specific metrics
    modified_duration: Optional[float] = None
    effective_duration: Optional[float] = None
    convexity: Optional[float] = None
    yield_to_maturity: Optional[float] = None
    credit_spread_duration: Optional[float] = None
    
    # Liquidity and concentration metrics
    liquidity_score: Optional[float] = None
    concentration_risk: Optional[float] = None
    sector_concentration: Optional[float] = None
    issuer_concentration: Optional[float] = None
    
    # Leverage and exposure metrics
    leverage_ratio: Optional[float] = None
    gross_exposure: Optional[float] = None
    net_exposure: Optional[float] = None
    
    # Historical metrics
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    # Metadata
    confidence_level: float = 0.95
    time_horizon_days: int = 1
    calculation_method: str = "HISTORICAL_SIMULATION"
    data_points: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskLimit:
    """Risk limit definition."""
    
    limit_id: str = field(default_factory=lambda: str(uuid4()))
    portfolio_id: str = None
    limit_type: LimitType = None
    limit_name: str = None
    limit_value: Union[float, Decimal] = None
    limit_currency: str = "INR"
    limit_unit: str = None  # %, absolute, etc.
    
    # Limit hierarchy
    parent_limit_id: Optional[str] = None
    child_limit_ids: List[str] = field(default_factory=list)
    
    # Monitoring parameters
    warning_threshold: float = 0.8  # 80% of limit
    critical_threshold: float = 0.95  # 95% of limit
    breach_threshold: float = 1.0  # 100% of limit
    
    # Status
    is_active: bool = True
    is_hard_limit: bool = True  # Hard vs soft limit
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    description: Optional[str] = None
    risk_owner: Optional[str] = None
    approval_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimitBreach:
    """Risk limit breach record."""
    
    breach_id: str = field(default_factory=lambda: str(uuid4()))
    limit_id: str = None
    portfolio_id: str = None
    breach_time: datetime = field(default_factory=datetime.utcnow)
    
    # Breach details
    limit_value: Union[float, Decimal] = None
    actual_value: Union[float, Decimal] = None
    breach_amount: Union[float, Decimal] = None
    breach_percentage: float = None
    
    # Breach classification
    breach_level: RiskLevel = RiskLevel.MEDIUM
    breach_type: str = "LIMIT_BREACH"  # LIMIT_BREACH, WARNING, CRITICAL
    
    # Status
    is_resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_method: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    # Notification
    notification_sent: bool = False
    notification_recipients: List[str] = field(default_factory=list)
    
    # Metadata
    triggered_by: Optional[str] = None
    market_conditions: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioPosition:
    """Portfolio position for risk analysis."""
    
    position_id: str = field(default_factory=lambda: str(uuid4()))
    portfolio_id: str = None
    bond_id: str = None
    participant_id: int = None
    
    # Position details
    quantity: Decimal = Decimal('0')
    face_value: Decimal = Decimal('0')
    market_value: Decimal = Decimal('0')
    cost_basis: Decimal = Decimal('0')
    
    # Risk metrics
    modified_duration: Optional[float] = None
    effective_duration: Optional[float] = None
    convexity: Optional[float] = None
    yield_to_maturity: Optional[float] = None
    credit_spread: Optional[float] = None
    
    # Market data
    current_price: Optional[Decimal] = None
    current_yield: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # P&L
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    total_pnl: Decimal = Decimal('0')
    
    # Metadata
    acquisition_date: Optional[date] = None
    maturity_date: Optional[date] = None
    rating: Optional[str] = None
    sector: Optional[str] = None
    issuer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    """Portfolio for risk management."""
    
    portfolio_id: str = field(default_factory=lambda: str(uuid4()))
    portfolio_name: str = None
    participant_id: int = None
    portfolio_type: str = "TRADING"  # TRADING, INVESTMENT, HEDGE
    
    # Portfolio details
    total_value: Decimal = Decimal('0')
    total_face_value: Decimal = Decimal('0')
    total_cost_basis: Decimal = Decimal('0')
    
    # Risk profile
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    investment_horizon: str = "MEDIUM_TERM"  # SHORT, MEDIUM, LONG
    target_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    description: Optional[str] = None
    risk_manager: Optional[str] = None
    compliance_officer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAlert:
    """Risk alert for monitoring and notification."""
    
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    portfolio_id: str = None
    alert_type: str = None
    alert_level: RiskLevel = RiskLevel.MEDIUM
    
    # Alert details
    title: str = None
    message: str = None
    alert_time: datetime = field(default_factory=datetime.utcnow)
    
    # Trigger conditions
    trigger_value: Union[float, Decimal] = None
    threshold_value: Union[float, Decimal] = None
    trigger_metric: str = None
    
    # Status
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Notification
    notification_sent: bool = False
    notification_recipients: List[str] = field(default_factory=list)
    escalation_level: int = 1
    
    # Metadata
    source_system: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    
    rule_id: str = field(default_factory=lambda: str(uuid4()))
    rule_name: str = None
    rule_type: str = None  # SEBI, RBI, INTERNAL
    
    # Rule parameters
    rule_description: str = None
    rule_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance checks
    check_frequency: str = "REAL_TIME"  # REAL_TIME, DAILY, WEEKLY, MONTHLY
    check_conditions: List[str] = field(default_factory=list)
    
    # Enforcement
    is_mandatory: bool = True
    enforcement_level: str = "HARD"  # HARD, SOFT, ADVISORY
    penalty_description: Optional[str] = None
    
    # Status
    is_active: bool = True
    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    regulatory_source: Optional[str] = None
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCheck:
    """Compliance check result."""
    
    check_id: str = field(default_factory=lambda: str(uuid4()))
    rule_id: str = None
    portfolio_id: str = None
    check_time: datetime = field(default_factory=datetime.utcnow)
    
    # Check results
    compliance_status: ComplianceStatus = ComplianceStatus.UNKNOWN
    check_result: bool = False
    violation_details: Optional[str] = None
    
    # Metrics
    actual_value: Union[float, Decimal] = None
    threshold_value: Union[float, Decimal] = None
    deviation: Union[float, Decimal] = None
    deviation_percentage: Optional[float] = None
    
    # Status
    is_resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_method: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    # Metadata
    check_method: str = "AUTOMATED"
    data_source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestScenario:
    """Stress test scenario definition."""
    
    scenario_id: str = field(default_factory=lambda: str(uuid4()))
    scenario_name: str = None
    scenario_type: str = None  # INTEREST_RATE, CREDIT, LIQUIDITY, MARKET
    
    # Scenario parameters
    description: str = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Stress levels
    stress_level: str = "MODERATE"  # MILD, MODERATE, SEVERE, EXTREME
    confidence_level: float = 0.95
    time_horizon_days: int = 10
    
    # Execution
    is_active: bool = True
    last_executed: Optional[datetime] = None
    execution_frequency: str = "WEEKLY"  # DAILY, WEEKLY, MONTHLY
    
    # Results
    expected_impact: Optional[float] = None
    worst_case_impact: Optional[float] = None
    probability: Optional[float] = None
    
    # Metadata
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Stress test execution result."""
    
    result_id: str = field(default_factory=lambda: str(uuid4()))
    scenario_id: str = None
    portfolio_id: str = None
    execution_time: datetime = field(default_factory=datetime.utcnow)
    
    # Results
    portfolio_value_before: Decimal = Decimal('0')
    portfolio_value_after: Decimal = Decimal('0')
    portfolio_value_change: Decimal = Decimal('0')
    portfolio_value_change_percent: float = 0.0
    
    # Risk metrics impact
    var_change: Optional[float] = None
    duration_change: Optional[float] = None
    convexity_change: Optional[float] = None
    
    # Position-level impacts
    position_impacts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis
    is_passed: bool = True
    failure_reason: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    execution_duration_seconds: float = 0.0
    data_points_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulatoryReport:
    """Regulatory report definition."""
    
    report_id: str = field(default_factory=lambda: str(uuid4()))
    report_name: str = None
    regulatory_body: str = None  # SEBI, RBI, etc.
    
    # Report details
    report_type: str = None
    reporting_frequency: str = None  # DAILY, WEEKLY, MONTHLY, QUARTERLY
    due_date: Optional[date] = None
    
    # Content
    report_data: Dict[str, Any] = field(default_factory=dict)
    report_format: str = "JSON"  # JSON, XML, CSV, PDF
    
    # Status
    status: str = "PENDING"  # PENDING, GENERATED, SUBMITTED, CONFIRMED, FAILED
    generation_time: Optional[datetime] = None
    submission_time: Optional[datetime] = None
    confirmation_time: Optional[datetime] = None
    
    # Submission
    submission_method: Optional[str] = None
    submission_reference: Optional[str] = None
    submission_status: Optional[str] = None
    error_details: Optional[str] = None
    
    # Metadata
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
