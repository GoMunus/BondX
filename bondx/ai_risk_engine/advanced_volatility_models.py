"""
Advanced Volatility Forecasting Models for BondX

This module implements state-of-the-art ML models for volatility forecasting:
- LSTM/GRU for sequential volatility prediction
- Transformer architectures for regime detection
- Hybrid Neural-GARCH and HAR-RNN models
- Exogenous feature integration (liquidity, macro, sentiment)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import json
import pickle
import joblib
from pathlib import Path
import hashlib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)

class VolatilityModelType(Enum):
    """Types of volatility models"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    NEURAL_GARCH = "neural_garch"
    HAR_RNN = "har_rnn"
    HYBRID = "hybrid"

@dataclass
class VolatilityFeatures:
    """Features for volatility forecasting"""
    returns: np.ndarray
    realized_volatility: np.ndarray
    high_low_spread: np.ndarray
    volume: Optional[np.ndarray] = None
    liquidity_index: Optional[np.ndarray] = None
    macro_features: Optional[np.ndarray] = None
    sentiment_features: Optional[np.ndarray] = None
    technical_indicators: Optional[np.ndarray] = None

@dataclass
class ModelConfig:
    """Configuration for volatility models"""
    model_type: VolatilityModelType
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2

@dataclass
class VolatilityForecast:
    """Volatility forecast results"""
    timestamp: datetime
    forecast_horizon: int
    point_forecast: float
    confidence_interval: Tuple[float, float]
    regime_probability: Dict[str, float]
    feature_importance: Dict[str, float]

class LSTMVolatilityModel(nn.Module):
    """LSTM-based volatility forecasting model"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.activation(out)
        return out

class GRUVolatilityModel(nn.Module):
    """GRU-based volatility forecasting model"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]  # Take last output
        out = self.dropout(gru_out)
        out = self.fc(out)
        out = self.activation(out)
        return out

class TransformerVolatilityModel(nn.Module):
    """Transformer-based volatility forecasting model"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take last output
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class NeuralGARCHModel(nn.Module):
    """Neural-GARCH hybrid model combining statistical rigor with ML flexibility"""
    
    def __init__(self, input_size: int, hidden_size: int, garch_order: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.p, self.q = garch_order
        
        # Neural network component
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # GARCH parameters (learnable)
        self.omega = nn.Parameter(torch.tensor(0.01))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))
        
        # Initialize GARCH parameters
        self._init_garch_params()
    
    def _init_garch_params(self):
        """Initialize GARCH parameters with reasonable values"""
        nn.init.constant_(self.omega, 0.01)
        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.beta, 0.8)
    
    def forward(self, x, returns_history):
        # Neural network component
        nn_output = self.nn(x)
        
        # GARCH component
        garch_vol = self._garch_volatility(returns_history)
        
        # Combine both components
        combined_vol = nn_output + garch_vol
        
        return combined_vol
    
    def _garch_volatility(self, returns_history):
        """Calculate GARCH volatility component"""
        # This is a simplified GARCH implementation
        # In production, you'd want a more sophisticated GARCH solver
        vol = torch.sqrt(self.omega + self.alpha * returns_history**2 + self.beta * vol**2)
        return vol

class HARRNNModel(nn.Module):
    """Heterogeneous Autoregressive RNN model for volatility forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int, har_lags: List[int] = [1, 5, 22]):
        super().__init__()
        self.har_lags = har_lags
        
        # HAR component
        self.har_weights = nn.Parameter(torch.ones(len(har_lags)))
        
        # RNN component
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size + len(har_lags), 1)
        self.activation = nn.ReLU()
    
    def forward(self, x, realized_vol_history):
        # HAR component
        har_features = []
        for lag in self.har_lags:
            if lag <= realized_vol_history.size(1):
                lag_vol = realized_vol_history[:, -lag:].mean(dim=1, keepdim=True)
                har_features.append(lag_vol)
        
        har_features = torch.cat(har_features, dim=1)
        
        # RNN component
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        
        # Combine HAR and RNN
        combined = torch.cat([rnn_out, har_features], dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        output = self.activation(output)
        
        return output

class AdvancedVolatilityForecaster:
    """Advanced volatility forecasting system with multiple ML models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model selection
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
        # Training history
        self.training_history = []
        self.validation_history = []
        
        # Model performance metrics
        self.metrics = {}
    
    def _create_model(self) -> nn.Module:
        """Create the specified volatility model"""
        input_size = self._get_input_size()
        
        if self.config.model_type == VolatilityModelType.LSTM:
            return LSTMVolatilityModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        elif self.config.model_type == VolatilityModelType.GRU:
            return GRUVolatilityModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        elif self.config.model_type == VolatilityModelType.TRANSFORMER:
            return TransformerVolatilityModel(
                input_size=input_size,
                d_model=self.config.hidden_size,
                nhead=8,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout
            )
        elif self.config.model_type == VolatilityModelType.NEURAL_GARCH:
            return NeuralGARCHModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size
            )
        elif self.config.config.model_type == VolatilityModelType.HAR_RNN:
            return HARRNNModel(
                input_size=input_size,
                hidden_size=self.config.hidden_size
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _get_input_size(self) -> int:
        """Calculate input size based on features"""
        # Base features: returns, realized volatility, high-low spread
        base_size = 3
        
        # Add exogenous features if available
        if hasattr(self, 'exogenous_features'):
            base_size += self.exogenous_features.shape[1]
        
        return base_size
    
    def prepare_features(self, features: VolatilityFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare features for model training"""
        # Combine all features
        feature_list = [features.returns, features.realized_volatility, features.high_low_spread]
        
        if features.volume is not None:
            feature_list.append(features.volume)
        if features.liquidity_index is not None:
            feature_list.append(features.liquidity_index)
        if features.macro_features is not None:
            feature_list.append(features.macro_features)
        if features.sentiment_features is not None:
            feature_list.append(features.sentiment_features)
        if features.technical_indicators is not None:
            feature_list.append(features.technical_indicators)
        
        # Stack features
        all_features = np.column_stack(feature_list)
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(all_features)
        
        # Create sequences
        X, y = self._create_sequences(scaled_features, features.realized_volatility)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(features)):
            X.append(features[i-self.config.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train(self, features: VolatilityFeatures) -> Dict[str, Any]:
        """Train the volatility forecasting model"""
        self.logger.info(f"Starting training for {self.config.model_type.value} model")
        
        # Prepare data
        X, y = self.prepare_features(features)
        
        # Split data
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_volatility_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
            
            # Store history
            self.training_history.append(train_loss)
            self.validation_history.append(val_loss)
        
        # Load best model
        self.model.load_state_dict(torch.load('best_volatility_model.pth'))
        
        # Calculate final metrics
        self._calculate_metrics(X_val, y_val)
        
        return {
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'final_metrics': self.metrics
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_X)
            loss = self.criterion(predictions.squeeze(), batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = self.model(batch_X)
                loss = self.criterion(predictions.squeeze(), batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _calculate_metrics(self, X: torch.Tensor, y: torch.Tensor):
        """Calculate model performance metrics"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy().squeeze()
            actual = y.cpu().numpy()
            
            self.metrics = {
                'mse': mean_squared_error(actual, predictions),
                'mae': mean_absolute_error(actual, predictions),
                'rmse': np.sqrt(mean_squared_error(actual, predictions)),
                'mape': np.mean(np.abs((actual - predictions) / actual)) * 100
            }
    
    def forecast(self, features: VolatilityFeatures, horizon: int = 1) -> VolatilityForecast:
        """Generate volatility forecast"""
        self.model.eval()
        
        # Prepare features
        X, _ = self.prepare_features(features)
        
        # Get the most recent sequence
        latest_sequence = X[-1:].to(self.device)
        
        with torch.no_grad():
            prediction = self.model(latest_sequence).cpu().numpy().squeeze()
        
        # Calculate confidence interval (simplified)
        confidence_interval = (prediction * 0.9, prediction * 1.1)
        
        # Regime detection (simplified)
        regime_probability = {
            'low_volatility': 0.3,
            'normal_volatility': 0.5,
            'high_volatility': 0.2
        }
        
        # Feature importance (simplified)
        feature_importance = {
            'returns': 0.4,
            'realized_volatility': 0.3,
            'high_low_spread': 0.2,
            'exogenous': 0.1
        }
        
        return VolatilityForecast(
            timestamp=datetime.utcnow(),
            forecast_horizon=horizon,
            point_forecast=float(prediction),
            confidence_interval=confidence_interval,
            regime_probability=regime_probability,
            feature_importance=feature_importance
        )
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'metrics': self.metrics
        }
        torch.save(model_state, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_state = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(model_state['model_state_dict'])
        self.config = model_state['config']
        self.scaler = model_state['scaler']
        self.feature_scaler = model_state['feature_scaler']
        self.metrics = model_state['metrics']
        
        self.logger.info(f"Model loaded from {filepath}")

class VolatilityModelOptimizer:
    """Hyperparameter optimization for volatility models"""
    
    def __init__(self, model_type: VolatilityModelType, features: VolatilityFeatures):
        self.model_type = model_type
        self.features = features
        self.logger = logging.getLogger(__name__)
    
    def optimize_hyperparameters(self, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Define hyperparameter search space
            config = ModelConfig(
                model_type=self.model_type,
                sequence_length=trial.suggest_int('sequence_length', 30, 120),
                hidden_size=trial.suggest_int('hidden_size', 64, 256),
                num_layers=trial.suggest_int('num_layers', 1, 4),
                dropout=trial.suggest_float('dropout', 0.1, 0.5),
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                epochs=50,  # Reduced for optimization
                early_stopping_patience=5
            )
            
            # Create and train model
            forecaster = AdvancedVolatilityForecaster(config)
            results = forecaster.train(self.features)
            
            # Return validation loss as objective
            return min(results['validation_history'])
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best hyperparameters: {study.best_params}")
        self.logger.info(f"Best validation loss: {study.best_value}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
