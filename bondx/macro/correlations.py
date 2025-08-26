"""
Cross-Market Correlation Engine

This module provides cross-market correlation analysis including:
- Rolling correlation matrices
- Macro factor correlations
- Near-PD enforcement
- Factor contribution analysis
- Shock linkage modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import cholesky, inv
from scipy.optimize import minimize

# ML imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf, OAS

logger = logging.getLogger(__name__)

class CorrelationMethod(Enum):
    """Available correlation calculation methods"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING = "rolling"
    EXPONENTIAL = "exponential"

class PDEnforcementMethod(Enum):
    """Methods for enforcing positive definiteness"""
    NEAREST_CORRELATION = "nearest_correlation"
    CHOLESKY = "cholesky"
    SHRINKAGE = "shrinkage"
    THRESHOLD = "threshold"

@dataclass
class MacroFactor:
    """Macroeconomic factor definition"""
    name: str
    category: str
    frequency: str  # daily, weekly, monthly
    last_value: float
    last_update: datetime
    volatility: float
    correlation_with_bonds: float
    impact_multiplier: float  # How much 1% change affects bond spreads

@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    condition_number: float
    is_positive_definite: bool
    pd_enforcement_method: Optional[str] = None
    correlation_method: str = "pearson"
    calculation_date: datetime = field(default_factory=datetime.now)

@dataclass
class ShockLinkage:
    """Shock linkage between macro factors and bond markets"""
    factor_name: str
    factor_shock: float  # Percentage change
    bond_sector: str
    spread_impact: float  # Basis points
    confidence: float
    lag_days: int
    calculation_date: datetime = field(default_factory=datetime.now)

class CrossMarketCorrelationEngine:
    """Engine for cross-market correlation analysis"""
    
    def __init__(self, seed: int = 42, window_size: int = 252):
        """Initialize correlation engine"""
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        
        # Macro factors (synthetic for demo)
        self.macro_factors = self._initialize_macro_factors()
        
        # Historical macro data
        self.macro_history = self._initialize_macro_history()
        
        # Sector mappings
        self.sector_mappings = self._initialize_sector_mappings()
        
        # Correlation matrices
        self.correlation_matrices = {}
        
        # Shock linkages
        self.shock_linkages = []
    
    def _initialize_macro_factors(self) -> Dict[str, MacroFactor]:
        """Initialize synthetic macroeconomic factors"""
        factors = {}
        now = datetime.now()
        
        # Oil price
        factors['oil_price'] = MacroFactor(
            name='Oil Price (WTI)',
            category='Commodity',
            frequency='daily',
            last_value=75.0,
            last_update=now,
            volatility=0.25,
            correlation_with_bonds=-0.3,
            impact_multiplier=0.3  # 1% oil change = 0.3 bps spread change
        )
        
        # USD Index
        factors['usd_index'] = MacroFactor(
            name='USD Index (DXY)',
            category='Currency',
            frequency='daily',
            last_value=100.0,
            last_update=now,
            volatility=0.12,
            correlation_with_bonds=0.2,
            impact_multiplier=0.2
        )
        
        # VIX
        factors['vix'] = MacroFactor(
            name='VIX Volatility Index',
            category='Volatility',
            frequency='daily',
            last_value=20.0,
            last_update=now,
            volatility=0.8,
            correlation_with_bonds=-0.4,
            impact_multiplier=0.5
        )
        
        # Treasury yields
        factors['treasury_10y'] = MacroFactor(
            name='10Y Treasury Yield',
            category='Interest Rate',
            frequency='daily',
            last_value=4.0,
            last_update=now,
            volatility=0.15,
            correlation_with_bonds=0.6,
            impact_multiplier=0.8
        )
        
        # Credit spreads
        factors['credit_spreads'] = MacroFactor(
            name='Investment Grade Credit Spreads',
            category='Credit',
            frequency='daily',
            last_value=120.0,
            last_update=now,
            volatility=0.3,
            correlation_with_bonds=0.7,
            impact_multiplier=1.0
        )
        
        # GDP growth
        factors['gdp_growth'] = MacroFactor(
            name='GDP Growth Rate',
            category='Economic',
            frequency='quarterly',
            last_value=2.5,
            last_update=now,
            volatility=0.5,
            correlation_with_bonds=-0.2,
            impact_multiplier=0.4
        )
        
        # Inflation
        factors['inflation'] = MacroFactor(
            name='CPI Inflation Rate',
            category='Economic',
            frequency='monthly',
            last_value=3.0,
            last_update=now,
            volatility=0.2,
            correlation_with_bonds=0.3,
            impact_multiplier=0.6
        )
        
        return factors
    
    def _initialize_macro_history(self) -> pd.DataFrame:
        """Initialize synthetic historical macro data"""
        # Generate 500 days of historical data
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        
        # Initialize with base values
        data = {
            'date': dates,
            'oil_price': 75.0,
            'usd_index': 100.0,
            'vix': 20.0,
            'treasury_10y': 4.0,
            'credit_spreads': 120.0,
            'gdp_growth': 2.5,
            'inflation': 3.0
        }
        
        df = pd.DataFrame(data)
        
        # Add random walks to simulate realistic movements
        for col in df.columns:
            if col != 'date':
                # Generate random walk
                random_walks = np.random.normal(0, 0.01, len(df))  # 1% daily volatility
                random_walks[0] = 0  # Start at base value
                
                # Cumulative sum for random walk
                cumulative_walks = np.cumsum(random_walks)
                
                # Apply to base value
                df[col] = df[col] * (1 + cumulative_walks)
                
                # Ensure positive values for certain factors
                if col in ['oil_price', 'usd_index', 'vix', 'treasury_10y', 'credit_spreads']:
                    df[col] = df[col].abs()
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        return df
    
    def _initialize_sector_mappings(self) -> Dict[str, Dict[str, float]]:
        """Initialize sector sensitivity to macro factors"""
        mappings = {
            'Energy': {
                'oil_price': 0.8,
                'usd_index': -0.3,
                'vix': -0.4,
                'treasury_10y': 0.2,
                'credit_spreads': 0.3,
                'gdp_growth': 0.4,
                'inflation': 0.2
            },
            'Financial': {
                'oil_price': 0.1,
                'usd_index': 0.4,
                'vix': -0.6,
                'treasury_10y': 0.7,
                'credit_spreads': 0.8,
                'gdp_growth': 0.5,
                'inflation': 0.3
            },
            'Technology': {
                'oil_price': 0.0,
                'usd_index': -0.2,
                'vix': -0.5,
                'treasury_10y': 0.3,
                'credit_spreads': 0.4,
                'gdp_growth': 0.6,
                'inflation': 0.1
            },
            'Healthcare': {
                'oil_price': 0.0,
                'usd_index': -0.1,
                'vix': -0.3,
                'treasury_10y': 0.2,
                'credit_spreads': 0.3,
                'gdp_growth': 0.3,
                'inflation': 0.2
            },
            'Consumer Goods': {
                'oil_price': 0.2,
                'usd_index': -0.2,
                'vix': -0.3,
                'treasury_10y': 0.2,
                'credit_spreads': 0.3,
                'gdp_growth': 0.5,
                'inflation': 0.4
            },
            'Utilities': {
                'oil_price': 0.3,
                'usd_index': -0.1,
                'vix': -0.2,
                'treasury_10y': 0.4,
                'credit_spreads': 0.3,
                'gdp_growth': 0.2,
                'inflation': 0.5
            },
            'Real Estate': {
                'oil_price': 0.1,
                'usd_index': -0.2,
                'vix': -0.4,
                'treasury_10y': 0.6,
                'credit_spreads': 0.5,
                'gdp_growth': 0.4,
                'inflation': 0.6
            },
            'Mining': {
                'oil_price': 0.6,
                'usd_index': -0.4,
                'vix': -0.5,
                'treasury_10y': 0.3,
                'credit_spreads': 0.4,
                'gdp_growth': 0.5,
                'inflation': 0.3
            },
            'Construction': {
                'oil_price': 0.2,
                'usd_index': -0.2,
                'vix': -0.3,
                'treasury_10y': 0.3,
                'credit_spreads': 0.4,
                'gdp_growth': 0.6,
                'inflation': 0.4
            },
            'Automotive': {
                'oil_price': 0.4,
                'usd_index': -0.3,
                'vix': -0.4,
                'treasury_10y': 0.3,
                'credit_spreads': 0.4,
                'gdp_growth': 0.5,
                'inflation': 0.3
            },
            'Infrastructure': {
                'oil_price': 0.2,
                'usd_index': -0.1,
                'vix': -0.3,
                'treasury_10y': 0.4,
                'credit_spreads': 0.4,
                'gdp_growth': 0.5,
                'inflation': 0.4
            },
            'Telecommunications': {
                'oil_price': 0.1,
                'usd_index': -0.2,
                'vix': -0.3,
                'treasury_10y': 0.3,
                'credit_spreads': 0.4,
                'gdp_growth': 0.4,
                'inflation': 0.3
            }
        }
        
        return mappings
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, 
                                  method: CorrelationMethod = CorrelationMethod.PEARSON,
                                  enforce_pd: bool = True,
                                  pd_method: PDEnforcementMethod = PDEnforcementMethod.NEAREST_CORRELATION) -> CorrelationResult:
        """Calculate correlation matrix from data"""
        self.logger.info(f"Calculating correlation matrix using {method.value}")
        
        # Remove non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric data found for correlation calculation")
        
        # Calculate correlations based on method
        if method == CorrelationMethod.PEARSON:
            corr_matrix = numeric_data.corr(method='pearson')
        elif method == CorrelationMethod.SPEARMAN:
            corr_matrix = numeric_data.corr(method='spearman')
        elif method == CorrelationMethod.KENDALL:
            corr_matrix = numeric_data.corr(method='kendall')
        elif method == CorrelationMethod.ROLLING:
            corr_matrix = self._calculate_rolling_correlation(numeric_data)
        elif method == CorrelationMethod.EXPONENTIAL:
            corr_matrix = self._calculate_exponential_correlation(numeric_data)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        # Check if matrix is positive definite
        eigenvalues = np.linalg.eigvals(corr_matrix.values)
        is_positive_definite = np.all(eigenvalues > 0)
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        
        # Enforce positive definiteness if requested
        pd_enforcement_method = None
        if enforce_pd and not is_positive_definite:
            corr_matrix = self._enforce_positive_definite(corr_matrix, pd_method)
            pd_enforcement_method = pd_method.value
            
            # Recalculate properties
            eigenvalues = np.linalg.eigvals(corr_matrix.values)
            is_positive_definite = np.all(eigenvalues > 0)
            condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        
        result = CorrelationResult(
            correlation_matrix=corr_matrix,
            eigenvalues=eigenvalues,
            condition_number=condition_number,
            is_positive_definite=is_positive_definite,
            pd_enforcement_method=pd_enforcement_method,
            correlation_method=method.value
        )
        
        # Store result
        self.correlation_matrices[method.value] = result
        
        return result
    
    def _calculate_rolling_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlation matrix"""
        # Use rolling window correlation
        rolling_corr = data.rolling(window=self.window_size, min_periods=30).corr()
        
        # Take the last correlation matrix
        last_corr = rolling_corr.iloc[-len(data.columns):, -len(data.columns):]
        
        return last_corr
    
    def _calculate_exponential_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate exponentially weighted correlation matrix"""
        # Use exponential weighting
        exp_corr = data.ewm(span=self.window_size).corr()
        
        # Take the last correlation matrix
        last_corr = exp_corr.iloc[-len(data.columns):, -len(data.columns):]
        
        return last_corr
    
    def _enforce_positive_definite(self, corr_matrix: pd.DataFrame, 
                                 method: PDEnforcementMethod) -> pd.DataFrame:
        """Enforce positive definiteness of correlation matrix"""
        self.logger.info(f"Enforcing positive definiteness using {method.value}")
        
        if method == PDEnforcementMethod.NEAREST_CORRELATION:
            return self._nearest_correlation_matrix(corr_matrix)
        elif method == PDEnforcementMethod.CHOLESKY:
            return self._cholesky_enforcement(corr_matrix)
        elif method == PDEnforcementMethod.SHRINKAGE:
            return self._shrinkage_enforcement(corr_matrix)
        elif method == PDEnforcementMethod.THRESHOLD:
            return self._threshold_enforcement(corr_matrix)
        else:
            raise ValueError(f"Unsupported PD enforcement method: {method}")
    
    def _nearest_correlation_matrix(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Find nearest correlation matrix using optimization"""
        # Convert to numpy array
        C = corr_matrix.values
        n = C.shape[0]
        
        # Objective function: minimize ||C - X||_F^2
        def objective(x):
            X = x.reshape(n, n)
            return np.sum((C - X) ** 2)
        
        # Constraints: diagonal = 1, symmetric, positive definite
        def constraint_diagonal(x):
            X = x.reshape(n, n)
            return np.diag(X) - 1
        
        def constraint_symmetric(x):
            X = x.reshape(n, n)
            return X - X.T
        
        # Initial guess: identity matrix
        x0 = np.eye(n).flatten()
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': constraint_diagonal},
            {'type': 'eq', 'fun': constraint_symmetric}
        ]
        
        # Bounds: correlations between -1 and 1
        bounds = [(-1, 1)] * (n * n)
        
        # Optimize
        result = minimize(objective, x0, constraints=constraints, bounds=bounds, method='SLSQP')
        
        if result.success:
            X = result.x.reshape(n, n)
            # Ensure positive definiteness by adjusting eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(X)
            eigenvalues = np.maximum(eigenvalues, 1e-6)  # Small positive threshold
            X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            return pd.DataFrame(X, index=corr_matrix.index, columns=corr_matrix.columns)
        else:
            self.logger.warning("Optimization failed, using fallback method")
            return self._threshold_enforcement(corr_matrix)
    
    def _cholesky_enforcement(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Enforce positive definiteness using Cholesky decomposition"""
        C = corr_matrix.values
        
        try:
            # Try Cholesky decomposition
            L = cholesky(C)
            C_pd = L @ L.T
            
            # Normalize to get correlation matrix
            D = np.diag(1 / np.sqrt(np.diag(C_pd)))
            C_corr = D @ C_pd @ D
            
            return pd.DataFrame(C_corr, index=corr_matrix.index, columns=corr_matrix.columns)
        except:
            self.logger.warning("Cholesky decomposition failed, using threshold method")
            return self._threshold_enforcement(corr_matrix)
    
    def _shrinkage_enforcement(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Enforce positive definiteness using shrinkage estimation"""
        C = corr_matrix.values
        
        # Use Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        lw.fit(C)
        C_shrink = lw.covariance_
        
        # Convert back to correlation matrix
        D = np.diag(1 / np.sqrt(np.diag(C_shrink)))
        C_corr = D @ C_shrink @ D
        
        return pd.DataFrame(C_corr, index=corr_matrix.index, columns=corr_matrix.columns)
    
    def _threshold_enforcement(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Enforce positive definiteness using eigenvalue thresholding"""
        C = corr_matrix.values
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Set negative eigenvalues to small positive value
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        
        # Reconstruct matrix
        C_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Normalize to get correlation matrix
        D = np.diag(1 / np.sqrt(np.diag(C_pd)))
        C_corr = D @ C_pd @ D
        
        return pd.DataFrame(C_corr, index=corr_matrix.index, columns=corr_matrix.columns)
    
    def calculate_macro_correlations(self, bond_data: pd.DataFrame, 
                                   sector_column: str = 'sector',
                                   spread_column: str = 'spread_bps') -> pd.DataFrame:
        """Calculate correlations between macro factors and bond spreads by sector"""
        self.logger.info("Calculating macro factor correlations by sector")
        
        # Get unique sectors
        sectors = bond_data[sector_column].unique()
        
        correlations = {}
        
        for sector in sectors:
            sector_data = bond_data[bond_data[sector_column] == sector]
            
            if len(sector_data) < 10:  # Need minimum data points
                continue
            
            sector_correlations = {}
            
            for factor_name, factor_data in self.macro_factors.items():
                if factor_name in self.macro_history.columns:
                    # Get factor data for the same period
                    factor_series = self.macro_history[factor_name].reindex(sector_data.index, method='ffill')
                    
                    # Calculate correlation
                    valid_data = pd.concat([sector_data[spread_column], factor_series], axis=1).dropna()
                    
                    if len(valid_data) >= 10:
                        correlation, p_value = pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
                        sector_correlations[factor_name] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'sample_size': len(valid_data)
                        }
            
            correlations[sector] = sector_correlations
        
        return pd.DataFrame(correlations).T
    
    def calculate_factor_contributions(self, bond_data: pd.DataFrame,
                                    sector_column: str = 'sector',
                                    spread_column: str = 'spread_bps') -> Dict[str, Dict[str, float]]:
        """Calculate factor contributions to bond spreads by sector"""
        self.logger.info("Calculating factor contributions by sector")
        
        factor_contributions = {}
        
        for sector in bond_data[sector_column].unique():
            sector_data = bond_data[bond_data[sector_column] == sector]
            
            if len(sector_data) < 10:
                continue
            
            sector_contributions = {}
            
            # Get sector sensitivity mappings
            sector_sensitivity = self.sector_mappings.get(sector, {})
            
            for factor_name, factor_data in self.macro_factors.items():
                if factor_name in sector_sensitivity:
                    # Calculate contribution based on sensitivity and impact multiplier
                    sensitivity = sector_sensitivity[factor_name]
                    impact = factor_data.impact_multiplier
                    
                    # Contribution = sensitivity * impact * factor volatility
                    contribution = sensitivity * impact * factor_data.volatility
                    
                    sector_contributions[factor_name] = contribution
            
            factor_contributions[sector] = sector_contributions
        
        return factor_contributions
    
    def model_shock_linkages(self, factor_shock: Dict[str, float],
                           bond_data: pd.DataFrame,
                           sector_column: str = 'sector') -> List[ShockLinkage]:
        """Model shock linkages between macro factors and bond markets"""
        self.logger.info("Modeling shock linkages")
        
        linkages = []
        
        for factor_name, shock_pct in factor_shock.items():
            if factor_name not in self.macro_factors:
                continue
            
            factor = self.macro_factors[factor_name]
            
            for sector in bond_data[sector_column].unique():
                sector_sensitivity = self.sector_mappings.get(sector, {})
                
                if factor_name in sector_sensitivity:
                    sensitivity = sector_sensitivity[factor_name]
                    impact = factor.impact_multiplier
                    
                    # Calculate spread impact in basis points
                    # Impact = shock_pct * sensitivity * impact_multiplier * 100 (to bps)
                    spread_impact = shock_pct * sensitivity * impact * 100
                    
                    # Calculate confidence based on historical correlation
                    confidence = min(0.95, max(0.1, abs(factor.correlation_with_bonds)))
                    
                    # Estimate lag (simplified)
                    lag_days = 1 if factor.frequency == 'daily' else 5
                    
                    linkage = ShockLinkage(
                        factor_name=factor_name,
                        factor_shock=shock_pct,
                        bond_sector=sector,
                        spread_impact=spread_impact,
                        confidence=confidence,
                        lag_days=lag_days
                    )
                    
                    linkages.append(linkage)
        
        # Store linkages
        self.shock_linkages.extend(linkages)
        
        return linkages
    
    def generate_macro_scenarios(self, scenario_type: str = "moderate") -> Dict[str, float]:
        """Generate macro shock scenarios"""
        scenarios = {
            "moderate": {
                'oil_price': 0.1,      # 10% increase
                'usd_index': 0.05,     # 5% increase
                'vix': 0.25,           # 25% increase
                'treasury_10y': 0.1,   # 10% increase
                'credit_spreads': 0.15, # 15% increase
                'gdp_growth': -0.1,    # 10% decrease
                'inflation': 0.1       # 10% increase
            },
            "severe": {
                'oil_price': 0.3,      # 30% increase
                'usd_index': 0.15,     # 15% increase
                'vix': 0.5,            # 50% increase
                'treasury_10y': 0.25,  # 25% increase
                'credit_spreads': 0.4, # 40% increase
                'gdp_growth': -0.25,   # 25% decrease
                'inflation': 0.25      # 25% increase
            },
            "extreme": {
                'oil_price': 0.5,      # 50% increase
                'usd_index': 0.25,     # 25% increase
                'vix': 1.0,            # 100% increase
                'treasury_10y': 0.5,   # 50% increase
                'credit_spreads': 0.75, # 75% increase
                'gdp_growth': -0.5,    # 50% decrease
                'inflation': 0.5       # 50% increase
            }
        }
        
        return scenarios.get(scenario_type, scenarios["moderate"])
    
    def calculate_portfolio_macro_exposure(self, bond_data: pd.DataFrame,
                                        sector_column: str = 'sector',
                                        value_column: str = 'issue_size') -> Dict[str, float]:
        """Calculate portfolio exposure to macro factors"""
        self.logger.info("Calculating portfolio macro exposure")
        
        # Group by sector and sum values
        sector_exposure = bond_data.groupby(sector_column)[value_column].sum()
        
        # Calculate factor exposure
        factor_exposure = {}
        
        for factor_name in self.macro_factors.keys():
            total_exposure = 0
            
            for sector, sector_value in sector_exposure.items():
                sector_sensitivity = self.sector_mappings.get(sector, {})
                factor_sensitivity = sector_sensitivity.get(factor_name, 0)
                
                # Exposure = sector_value * factor_sensitivity
                total_exposure += sector_value * factor_sensitivity
            
            factor_exposure[factor_name] = total_exposure
        
        return factor_exposure
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of all correlation matrices"""
        summary = {}
        
        for method, result in self.correlation_matrices.items():
            summary[method] = {
                'is_positive_definite': result.is_positive_definite,
                'condition_number': result.condition_number,
                'pd_enforcement_method': result.pd_enforcement_method,
                'calculation_date': result.calculation_date.isoformat(),
                'matrix_shape': result.correlation_matrix.shape
            }
        
        return summary
    
    def save_correlation_data(self, filepath: str):
        """Save correlation data to file"""
        data_to_save = {
            'correlation_matrices': {
                method: {
                    'matrix': result.correlation_matrix.to_dict(),
                    'eigenvalues': result.eigenvalues.tolist(),
                    'condition_number': result.condition_number,
                    'is_positive_definite': result.is_positive_definite,
                    'pd_enforcement_method': result.pd_enforcement_method,
                    'correlation_method': result.correlation_method,
                    'calculation_date': result.calculation_date.isoformat()
                }
                for method, result in self.correlation_matrices.items()
            },
            'shock_linkages': [
                {
                    'factor_name': link.factor_name,
                    'factor_shock': link.factor_shock,
                    'bond_sector': link.bond_sector,
                    'spread_impact': link.spread_impact,
                    'confidence': link.confidence,
                    'lag_days': link.lag_days,
                    'calculation_date': link.calculation_date.isoformat()
                }
                for link in self.shock_linkages
            ],
            'macro_factors': {
                name: {
                    'category': factor.category,
                    'frequency': factor.frequency,
                    'last_value': factor.last_value,
                    'last_update': factor.last_update.isoformat(),
                    'volatility': factor.volatility,
                    'correlation_with_bonds': factor.correlation_with_bonds,
                    'impact_multiplier': factor.impact_multiplier
                }
                for name, factor in self.macro_factors.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        self.logger.info(f"Correlation data saved to {filepath}")
    
    def load_correlation_data(self, filepath: str):
        """Load correlation data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load correlation matrices
            for method, matrix_data in data.get('correlation_matrices', {}).items():
                corr_matrix = pd.DataFrame(matrix_data['matrix'])
                eigenvalues = np.array(matrix_data['eigenvalues'])
                
                result = CorrelationResult(
                    correlation_matrix=corr_matrix,
                    eigenvalues=eigenvalues,
                    condition_number=matrix_data['condition_number'],
                    is_positive_definite=matrix_data['is_positive_definite'],
                    pd_enforcement_method=matrix_data.get('pd_enforcement_method'),
                    correlation_method=matrix_data['correlation_method'],
                    calculation_date=datetime.fromisoformat(matrix_data['calculation_date'])
                )
                
                self.correlation_matrices[method] = result
            
            # Load shock linkages
            self.shock_linkages = []
            for link_data in data.get('shock_linkages', []):
                link = ShockLinkage(
                    factor_name=link_data['factor_name'],
                    factor_shock=link_data['factor_shock'],
                    bond_sector=link_data['bond_sector'],
                    spread_impact=link_data['spread_impact'],
                    confidence=link_data['confidence'],
                    lag_days=link_data['lag_days'],
                    calculation_date=datetime.fromisoformat(link_data['calculation_date'])
                )
                self.shock_linkages.append(link)
            
            self.logger.info(f"Correlation data loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load correlation data: {e}")
            raise
