"""
Currency Handler

This module provides multi-currency support including:
- FX conversion between USD, EUR, INR
- Yield curve normalization
- Currency-specific conventions
- Day count and pricing adjustments
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

logger = logging.getLogger(__name__)

class Currency(Enum):
    """Supported currencies"""
    USD = "USD"
    EUR = "EUR"
    INR = "INR"

class DayCountConvention(Enum):
    """Day count conventions"""
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"
    THIRTY_365 = "30/365"

@dataclass
class CurrencyConfig:
    """Configuration for a currency"""
    code: Currency
    name: str
    symbol: str
    day_count: DayCountConvention
    business_days: List[int]  # 0=Monday, 6=Sunday
    settlement_days: int
    base_rate: float  # Base interest rate
    volatility: float  # FX volatility
    correlation_matrix: Dict[str, float] = field(default_factory=dict)

@dataclass
class FXRate:
    """Foreign exchange rate"""
    from_currency: Currency
    to_currency: Currency
    rate: float
    timestamp: datetime
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

@dataclass
class YieldCurve:
    """Yield curve for a currency"""
    currency: Currency
    tenors: List[float]  # in years
    rates: List[float]  # in decimal
    day_count: DayCountConvention
    interpolation_method: str = "linear"
    last_update: Optional[datetime] = None

class CurrencyHandler:
    """Handles multi-currency operations and FX conversions"""
    
    def __init__(self, base_currency: Currency = Currency.USD, seed: int = 42):
        """Initialize currency handler"""
        self.base_currency = base_currency
        self.seed = seed
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
        
        # Initialize currency configurations
        self.currency_configs = self._initialize_currency_configs()
        
        # FX rates (synthetic for demo)
        self.fx_rates = self._initialize_fx_rates()
        
        # Yield curves (synthetic for demo)
        self.yield_curves = self._initialize_yield_curves()
        
        # Historical FX data
        self.fx_history = self._initialize_fx_history()
        
        # Correlation matrix
        self.correlation_matrix = self._initialize_correlation_matrix()
    
    def _initialize_currency_configs(self) -> Dict[Currency, CurrencyConfig]:
        """Initialize currency configurations"""
        configs = {}
        
        # USD configuration
        configs[Currency.USD] = CurrencyConfig(
            code=Currency.USD,
            name="US Dollar",
            symbol="$",
            day_count=DayCountConvention.ACT_360,
            business_days=[0, 1, 2, 3, 4],  # Monday to Friday
            settlement_days=2,
            base_rate=0.05,  # 5% base rate
            volatility=0.15,  # 15% FX volatility
            correlation_matrix={
                "EUR": 0.7,
                "INR": 0.3
            }
        )
        
        # EUR configuration
        configs[Currency.EUR] = CurrencyConfig(
            code=Currency.EUR,
            name="Euro",
            symbol="€",
            day_count=DayCountConvention.ACT_360,
            business_days=[0, 1, 2, 3, 4],
            settlement_days=2,
            base_rate=0.03,  # 3% base rate
            volatility=0.12,  # 12% FX volatility
            correlation_matrix={
                "USD": 0.7,
                "INR": 0.4
            }
        )
        
        # INR configuration
        configs[Currency.INR] = CurrencyConfig(
            code=Currency.INR,
            name="Indian Rupee",
            symbol="₹",
            day_count=DayCountConvention.ACT_365,
            business_days=[0, 1, 2, 3, 4],
            settlement_days=1,
            base_rate=0.06,  # 6% base rate
            volatility=0.20,  # 20% FX volatility
            correlation_matrix={
                "USD": 0.3,
                "EUR": 0.4
            }
        )
        
        return configs
    
    def _initialize_fx_rates(self) -> Dict[Tuple[Currency, Currency], FXRate]:
        """Initialize synthetic FX rates"""
        rates = {}
        now = datetime.now()
        
        # USD/EUR
        rates[(Currency.USD, Currency.EUR)] = FXRate(
            from_currency=Currency.USD,
            to_currency=Currency.EUR,
            rate=0.85,
            timestamp=now,
            source="synthetic",
            bid=0.8495,
            ask=0.8505,
            spread=0.001
        )
        
        # EUR/USD
        rates[(Currency.EUR, Currency.USD)] = FXRate(
            from_currency=Currency.EUR,
            to_currency=Currency.USD,
            rate=1.176,
            timestamp=now,
            source="synthetic",
            bid=1.1755,
            ask=1.1765,
            spread=0.001
        )
        
        # USD/INR
        rates[(Currency.USD, Currency.INR)] = FXRate(
            from_currency=Currency.USD,
            to_currency=Currency.INR,
            rate=75.0,
            timestamp=now,
            source="synthetic",
            bid=74.95,
            ask=75.05,
            spread=0.1
        )
        
        # INR/USD
        rates[(Currency.INR, Currency.USD)] = FXRate(
            from_currency=Currency.INR,
            to_currency=Currency.USD,
            rate=0.0133,
            timestamp=now,
            source="synthetic",
            bid=0.01325,
            ask=0.01335,
            spread=0.0001
        )
        
        # EUR/INR
        rates[(Currency.EUR, Currency.INR)] = FXRate(
            from_currency=Currency.EUR,
            to_currency=Currency.INR,
            rate=88.2,
            timestamp=now,
            source="synthetic",
            bid=88.15,
            ask=88.25,
            spread=0.1
        )
        
        # INR/EUR
        rates[(Currency.INR, Currency.EUR)] = FXRate(
            from_currency=Currency.INR,
            to_currency=Currency.EUR,
            rate=0.0113,
            timestamp=now,
            source="synthetic",
            bid=0.01125,
            ask=0.01135,
            spread=0.0001
        )
        
        return rates
    
    def _initialize_yield_curves(self) -> Dict[Currency, YieldCurve]:
        """Initialize synthetic yield curves"""
        curves = {}
        
        # USD yield curve
        curves[Currency.USD] = YieldCurve(
            currency=Currency.USD,
            tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
            rates=[0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065],
            day_count=DayCountConvention.ACT_360,
            interpolation_method="cubic",
            last_update=datetime.now()
        )
        
        # EUR yield curve
        curves[Currency.EUR] = YieldCurve(
            currency=Currency.EUR,
            tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
            rates=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055],
            day_count=DayCountConvention.ACT_360,
            interpolation_method="cubic",
            last_update=datetime.now()
        )
        
        # INR yield curve
        curves[Currency.INR] = YieldCurve(
            currency=Currency.INR,
            tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
            rates=[0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085],
            day_count=DayCountConvention.ACT_365,
            interpolation_method="cubic",
            last_update=datetime.now()
        )
        
        return curves
    
    def _initialize_fx_history(self) -> Dict[Tuple[Currency, Currency], List[FXRate]]:
        """Initialize synthetic FX history"""
        history = {}
        now = datetime.now()
        
        # Generate 100 days of historical data
        for (from_curr, to_curr), base_rate in self.fx_rates.items():
            rates = []
            for i in range(100):
                # Add some random walk to base rate
                random_walk = np.random.normal(0, 0.01)  # 1% daily volatility
                new_rate = base_rate.rate * (1 + random_walk)
                
                # Ensure rate is positive
                new_rate = max(new_rate, 0.001)
                
                # Create historical rate
                hist_rate = FXRate(
                    from_currency=from_curr,
                    to_currency=to_curr,
                    rate=new_rate,
                    timestamp=now - timedelta(days=100-i),
                    source="synthetic"
                )
                rates.append(hist_rate)
            
            history[(from_curr, to_curr)] = rates
        
        return history
    
    def _initialize_correlation_matrix(self) -> pd.DataFrame:
        """Initialize currency correlation matrix"""
        currencies = [curr.value for curr in Currency]
        correlation_data = np.array([
            [1.0, 0.7, 0.3],
            [0.7, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        
        return pd.DataFrame(
            correlation_data,
            index=currencies,
            columns=currencies
        )
    
    def get_fx_rate(self, from_currency: Currency, to_currency: Currency) -> Optional[FXRate]:
        """Get current FX rate between two currencies"""
        if from_currency == to_currency:
            return FXRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=1.0,
                timestamp=datetime.now(),
                source="identity"
            )
        
        key = (from_currency, to_currency)
        if key in self.fx_rates:
            return self.fx_rates[key]
        
        # Try reverse rate
        reverse_key = (to_currency, from_currency)
        if reverse_key in self.fx_rates:
            reverse_rate = self.fx_rates[reverse_key]
            return FXRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=1.0 / reverse_rate.rate,
                timestamp=reverse_rate.timestamp,
                source=reverse_rate.source,
                bid=1.0 / reverse_rate.ask if reverse_rate.ask else None,
                ask=1.0 / reverse_rate.bid if reverse_rate.bid else None,
                spread=reverse_rate.spread
            )
        
        return None
    
    def convert_currency(self, amount: float, from_currency: Currency, 
                        to_currency: Currency, use_mid_rate: bool = True) -> Optional[float]:
        """Convert amount between currencies"""
        if from_currency == to_currency:
            return amount
        
        fx_rate = self.get_fx_rate(from_currency, to_currency)
        if not fx_rate:
            return None
        
        if use_mid_rate:
            return amount * fx_rate.rate
        else:
            # Use bid/ask rates for more realistic conversion
            if fx_rate.bid and fx_rate.ask:
                # For buy (from -> to), use ask rate
                # For sell (to -> from), use bid rate
                # Here we'll use mid rate as default
                return amount * fx_rate.rate
            else:
                return amount * fx_rate.rate
    
    def get_yield_curve(self, currency: Currency) -> Optional[YieldCurve]:
        """Get yield curve for a currency"""
        return self.yield_curves.get(currency)
    
    def interpolate_rate(self, currency: Currency, tenor: float) -> Optional[float]:
        """Interpolate rate for a specific tenor"""
        curve = self.get_yield_curve(currency)
        if not curve:
            return None
        
        # Simple linear interpolation
        tenors = np.array(curve.tenors)
        rates = np.array(curve.rates)
        
        if tenor <= tenors.min():
            return rates[0]
        elif tenor >= tenors.max():
            return rates[-1]
        else:
            # Find interpolation points
            idx = np.searchsorted(tenors, tenor)
            if idx == 0:
                return rates[0]
            elif idx == len(tenors):
                return rates[-1]
            else:
                # Linear interpolation
                t1, t2 = tenors[idx-1], tenors[idx]
                r1, r2 = rates[idx-1], rates[idx]
                return r1 + (r2 - r1) * (tenor - t1) / (t2 - t1)
    
    def calculate_day_count_fraction(self, start_date: datetime, end_date: datetime, 
                                   currency: Currency) -> float:
        """Calculate day count fraction for a currency"""
        config = self.currency_configs.get(currency)
        if not config:
            return 0.0
        
        # Calculate actual days
        days = (end_date - start_date).days
        
        if config.day_count == DayCountConvention.ACT_360:
            return days / 360.0
        elif config.day_count == DayCountConvention.ACT_365:
            return days / 365.0
        elif config.day_count == DayCountConvention.ACT_ACT:
            # Simplified ACT/ACT calculation
            return days / 365.0
        elif config.day_count == DayCountConvention.THIRTY_360:
            # Simplified 30/360 calculation
            return days / 360.0
        elif config.day_count == DayCountConvention.THIRTY_365:
            return days / 365.0
        else:
            return days / 365.0  # Default
    
    def normalize_to_base_currency(self, df: pd.DataFrame, 
                                 price_column: str = 'price',
                                 currency_column: str = 'currency') -> pd.DataFrame:
        """Normalize prices to base currency"""
        df_normalized = df.copy()
        
        # Add normalized price column
        df_normalized[f'{price_column}_base'] = df_normalized.apply(
            lambda row: self.convert_currency(
                row[price_column],
                Currency(row[currency_column]),
                self.base_currency
            ) if pd.notna(row[price_column]) and pd.notna(row[currency_column]) else None,
            axis=1
        )
        
        return df_normalized
    
    def calculate_cross_currency_spread(self, base_currency: Currency, 
                                      target_currency: Currency) -> Optional[float]:
        """Calculate cross-currency basis spread"""
        # Get yield curves
        base_curve = self.get_yield_curve(base_currency)
        target_curve = self.get_yield_curve(target_currency)
        
        if not base_curve or not target_curve:
            return None
        
        # Calculate 5-year spread as example
        base_5y = self.interpolate_rate(base_currency, 5.0)
        target_5y = self.interpolate_rate(target_currency, 5.0)
        
        if base_5y is None or target_5y is None:
            return None
        
        # Cross-currency basis = target_rate - base_rate - fx_forward_rate
        fx_rate = self.get_fx_rate(base_currency, target_currency)
        if not fx_rate:
            return None
        
        # Simplified calculation (in practice, this would involve forward FX rates)
        basis_spread = target_5y - base_5y
        
        return basis_spread
    
    def generate_fx_scenario(self, base_currency: Currency, 
                           shock_size: float = 0.1) -> Dict[Currency, float]:
        """Generate FX shock scenario"""
        scenario = {}
        
        for currency in Currency:
            if currency == base_currency:
                scenario[currency] = 1.0
            else:
                # Generate random shock
                shock = np.random.normal(0, shock_size)
                scenario[currency] = 1.0 + shock
        
        return scenario
    
    def apply_fx_shock(self, df: pd.DataFrame, shock_scenario: Dict[Currency, float],
                       price_column: str = 'price',
                       currency_column: str = 'currency') -> pd.DataFrame:
        """Apply FX shock to portfolio"""
        df_shocked = df.copy()
        
        # Add shocked price column
        df_shocked[f'{price_column}_shocked'] = df_shocked.apply(
            lambda row: self._apply_single_fx_shock(
                row[price_column],
                row[currency_column],
                shock_scenario
            ) if pd.notna(row[price_column]) and pd.notna(row[currency_column]) else None,
            axis=1
        )
        
        return df_shocked
    
    def _apply_single_fx_shock(self, price: float, currency: str, 
                              shock_scenario: Dict[Currency, float]) -> float:
        """Apply FX shock to a single price"""
        try:
            currency_enum = Currency(currency)
            shock_factor = shock_scenario.get(currency_enum, 1.0)
            return price * shock_factor
        except (ValueError, KeyError):
            return price
    
    def get_currency_volatility(self, currency: Currency) -> float:
        """Get volatility for a currency"""
        config = self.currency_configs.get(currency)
        return config.volatility if config else 0.15
    
    def calculate_portfolio_fx_exposure(self, df: pd.DataFrame,
                                      value_column: str = 'market_value',
                                      currency_column: str = 'currency') -> Dict[Currency, float]:
        """Calculate portfolio FX exposure by currency"""
        exposure = {}
        
        for currency in Currency:
            currency_df = df[df[currency_column] == currency.value]
            total_value = currency_df[value_column].sum()
            exposure[currency] = total_value
        
        return exposure
    
    def hedge_fx_exposure(self, exposure: Dict[Currency, float],
                          target_currency: Currency = None) -> Dict[Currency, float]:
        """Calculate FX hedge amounts"""
        if target_currency is None:
            target_currency = self.base_currency
        
        hedge_amounts = {}
        
        for currency, amount in exposure.items():
            if currency != target_currency:
                # Convert to target currency
                converted_amount = self.convert_currency(amount, currency, target_currency)
                if converted_amount is not None:
                    hedge_amounts[currency] = -converted_amount  # Negative for hedge
        
        return hedge_amounts
    
    def save_configuration(self, filepath: str):
        """Save currency configuration to file"""
        config_data = {
            'base_currency': self.base_currency.value,
            'seed': self.seed,
            'currency_configs': {
                curr.value: {
                    'name': config.name,
                    'symbol': config.symbol,
                    'day_count': config.day_count.value,
                    'business_days': config.business_days,
                    'settlement_days': config.settlement_days,
                    'base_rate': config.base_rate,
                    'volatility': config.volatility
                }
                for curr, config in self.currency_configs.items()
            },
            'fx_rates': {
                f"{from_curr.value}_{to_curr.value}": {
                    'rate': rate.rate,
                    'timestamp': rate.timestamp.isoformat(),
                    'bid': rate.bid,
                    'ask': rate.ask,
                    'spread': rate.spread
                }
                for (from_curr, to_curr), rate in self.fx_rates.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        self.logger.info(f"Currency configuration saved to {filepath}")
    
    def load_configuration(self, filepath: str):
        """Load currency configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update base currency
            self.base_currency = Currency(config_data['base_currency'])
            
            # Update seed
            self.seed = config_data.get('seed', 42)
            np.random.seed(self.seed)
            
            # Update FX rates
            for key, rate_data in config_data.get('fx_rates', {}).items():
                from_curr, to_curr = key.split('_')
                from_currency = Currency(from_curr)
                to_currency = Currency(to_curr)
                
                self.fx_rates[(from_currency, to_currency)] = FXRate(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    rate=rate_data['rate'],
                    timestamp=datetime.fromisoformat(rate_data['timestamp']),
                    source="loaded",
                    bid=rate_data.get('bid'),
                    ask=rate_data.get('ask'),
                    spread=rate_data.get('spread')
                )
            
            self.logger.info(f"Currency configuration loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load currency configuration: {e}")
            raise
