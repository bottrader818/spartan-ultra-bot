import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
from dataclasses import asdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from .base_strategy import BaseStrategy, TradingSignal, SignalDirection


class AIDataset(Dataset):
    """PyTorch Dataset for AI model training/inference"""
    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


class SignalFusionModel(nn.Module):
    """Neural network for signal fusion"""
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x))


class AISignalFusionStrategy(BaseStrategy):
    """
    AI-Powered Signal Fusion Strategy
    Combines technical indicators and ML for adaptive, confidence-scored signals
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_path = config.get('model_path')
        self.input_features = config.get('input_features', [
            'rsi', 'macd', 'volume_zscore', 'volatility', 'sma_distance'
        ])
        self.train_interval = config.get('train_interval', 100)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.warmup_period = config.get('warmup_period', 200)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SignalFusionModel(len(self.input_features)).to(self.device)
        self.scaler = StandardScaler()
        self.is_model_trained = False

        self.feature_buffer: List[np.ndarray] = []
        self.target_buffer: List[float] = []
        self.signal_history: List[TradingSignal] = []
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'avg_holding_period': timedelta(0),
            'model_accuracy': 0.0
        }

        if self.model_path:
            self._load_model()
        else:
            self.logger.info("Initializing new signal fusion model")

        self.logger.info(f"AI Signal Fusion Strategy initialized with config: {config}")    def _load_model(self):
        """Load pre-trained model from file"""
        try:
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state['model'])
            self.scaler = state['scaler']
            self.is_model_trained = True
            self.logger.info(f"Loaded pre-trained model from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.is_model_trained = False

    def _save_model(self, path: str):
        """Save current model state to file"""
        try:
            torch.save({
                'model': self.model.state_dict(),
                'scaler': self.scaler
            }, path)
            self.logger.info(f"Saved model to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}", exc_info=True)

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """
        Generate AI-powered trading signals
        """
        signals = []

        for symbol, df in data.items():
            try:
                if len(df) < self.warmup_period:
                    continue

                features = self._extract_features(symbol, df)
                if features is None:
                    continue

                confidence = self._predict(features)
                if confidence >= 0.65:
                    signal = self._create_signal(symbol, df, confidence)
                    signals.append(signal)

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)

        return signals

    def _extract_features(self, symbol: str, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract and normalize features for AI model"""
        try:
            features = {
                'rsi': self._calculate_rsi(df['close'])[-1],
                'macd': self._calculate_macd(df['close'])[-1],
                'volume_zscore': self._calculate_volume_zscore(df['volume'])[-1],
                'volatility': self._calculate_volatility(df['close'])[-1],
                'sma_distance': self._calculate_sma_distance(df['close'])[-1]
            }

            feature_vector = np.array([features[f] for f in self.input_features]).reshape(1, -1)

            if self.is_model_trained:
                return self.scaler.transform(feature_vector)
            return feature_vector

        except Exception as e:
            self.logger.error(f"Feature extraction failed for {symbol}: {str(e)}")
            return None

    def _predict(self, features: np.ndarray) -> float:
        """Get confidence prediction from AI model"""
        try:
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                output = self.model(features_tensor)
                return output.item()
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return 0.0    def _create_signal(self, symbol: str, df: pd.DataFrame, confidence: float) -> TradingSignal:
        """Create trading signal with AI-powered parameters"""
        current_price = df['close'].iloc[-1]
        atr = self._calculate_atr(df)[-1]

        direction = (
            SignalDirection.LONG
            if current_price > df['close'].rolling(20).mean().iloc[-1]
            else SignalDirection.SHORT
        )

        position_size = self._calculate_position_size(current_price, atr, confidence)

        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=current_price - (atr * 2) if direction == SignalDirection.LONG
            else current_price + (atr * 2),
            take_profit=current_price + (atr * 3) if direction == SignalDirection.LONG
            else current_price - (atr * 3),
            confidence=confidence,
            size=position_size,
            rationale=f"AI Signal Fusion (Confidence: {confidence:.2%})",
            metadata={
                'strategy': 'AI Fusion',
                'model_version': '1.0',
                'features': {f: df[f].iloc[-1] for f in self.input_features}
            }
        )

        self._update_signal_history(signal)
        return signal

    def _update_signal_history(self, signal: TradingSignal):
        """Update signal history and performance metrics"""
        self.signal_history.append(signal)
        self.performance_metrics['total_signals'] += 1

        self.logger.info(
            f"Generated {signal.direction.name} signal for {signal.symbol} "
            f"with {signal.confidence:.2%} confidence"
        )

    def on_trade_outcome(self, symbol: str, pnl: float, exit_time: datetime):
        """Update model with trade outcome for learning"""
        try:
            signal = next((s for s in reversed(self.signal_history) if s.symbol == symbol), None)
            if not signal:
                return

            features = self._get_features_for_signal(signal)
            if features is not None:
                self.feature_buffer.append(features)
                self.target_buffer.append(1.0 if pnl > 0 else 0.0)

                if len(self.feature_buffer) % self.train_interval == 0:
                    self._retrain_model()

        except Exception as e:
            self.logger.error(f"Error processing trade outcome: {str(e)}", exc_info=True)

    def _get_features_for_signal(self, signal: TradingSignal) -> Optional[np.ndarray]:
        """Reconstruct features from signal metadata"""
        try:
            return np.array([signal.metadata['features'][f] for f in self.input_features])
        except Exception:
            return None    def _retrain_model(self):
        """Retrain model with new data"""
        try:
            if len(self.feature_buffer) < 100:
                return

            X = np.array(self.feature_buffer)
            y = np.array(self.target_buffer)

            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            dataset = AIDataset(X_scaled, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            self.model.train()
            for _ in range(10):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X.to(self.device)).squeeze()
                    loss = criterion(outputs, batch_y.to(self.device))
                    loss.backward()
                    optimizer.step()

            self.is_model_trained = True
            self.logger.info(f"Retrained model with {len(X)} samples")

        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}", exc_info=True)

    # === Technical Indicator Placeholders ===
    def _calculate_rsi(self, prices: pd.Series) -> np.ndarray: ...
    def _calculate_macd(self, prices: pd.Series) -> np.ndarray: ...
    def _calculate_volume_zscore(self, volumes: pd.Series) -> np.ndarray: ...
    def _calculate_volatility(self, prices: pd.Series) -> np.ndarray: ...
    def _calculate_sma_distance(self, prices: pd.Series) -> np.ndarray: ...
    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray: ...
    def _calculate_position_size(self, price: float, atr: float, confidence: float) -> float: ...

    def save_state(self, path: str):
        """Save strategy state including model"""
        try:
            state = {
                'feature_buffer': self.feature_buffer,
                'target_buffer': self.target_buffer,
                'signal_history': [asdict(s) for s in self.signal_history[-100:]],
                'performance_metrics': self.performance_metrics,
                'config': {
                    'input_features': self.input_features,
                    'train_interval': self.train_interval,
                    'max_position_size': self.max_position_size,
                    'risk_per_trade': self.risk_per_trade,
                    'warmup_period': self.warmup_period
                }
            }

            torch.save(state, path)
            self._save_model(f"{path}.pt")
            self.logger.info(f"Saved strategy state to {path}")

        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}", exc_info=True)

    def load_state(self, path: str):
        """Load strategy state including model"""
        try:
            state = torch.load(path, map_location=self.device)

            self.feature_buffer = state.get('feature_buffer', [])
            self.target_buffer = state.get('target_buffer', [])
            self.signal_history = [TradingSignal(**s) for s in state.get('signal_history', [])]
            self.performance_metrics = state.get('performance_metrics', {})

            config = state.get('config', {})
            self.input_features = config.get('input_features', self.input_features)
            self.train_interval = config.get('train_interval', self.train_interval)
            self.max_position_size = config.get('max_position_size', self.max_position_size)
            self.risk_per_trade = config.get('risk_per_trade', self.risk_per_trade)
            self.warmup_period = config.get('warmup_period', self.warmup_period)

            self._load_model(f"{path}.pt")
            self.logger.info(f"Loaded strategy state from {path}")

        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}", exc_info=True)
