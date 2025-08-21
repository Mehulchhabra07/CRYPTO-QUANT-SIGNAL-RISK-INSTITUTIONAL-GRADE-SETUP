"""
Enhanced Feature Engineering for Hedge-Fund Grade Analysis
Implements advanced features for microstructure, derivatives, cross-asset signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngine:
    """
    Advanced feature engineering for crypto trading with hedge-fund grade indicators.
    """
    
    def __init__(self):
        self.feature_cache = {}
        
    def compute_microstructure_features(self, df: pd.DataFrame, orderbook_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compute microstructure features from OHLCV and optional orderbook data.
        """
        features = df.copy()
        
        # Basic spread estimation from OHLCV
        features['estimated_spread_pct'] = 100 * (features['High'] - features['Low']) / features['Close']
        
        # Trade imbalance proxy using volume-weighted price movements
        features['price_impact'] = np.log(features['Close'] / features['Open'])
        features['volume_impact'] = features['Volume'] / features['Volume'].rolling(20).mean()
        
        # Intrabar momentum (proxy for aggressive vs passive flow)
        features['intrabar_momentum'] = (features['Close'] - features['Open']) / (features['High'] - features['Low'] + 1e-9)
        
        # Volume profile indicators
        features['volume_acceleration'] = features['Volume'].pct_change()
        features['volume_zscore'] = self._zscore(features['Volume'], 20)
        
        # Tick direction proxy (simplified without tick data)
        features['price_direction'] = np.sign(features['Close'].diff())
        features['volume_weighted_direction'] = features['price_direction'] * features['Volume']
        features['net_buying_pressure'] = features['volume_weighted_direction'].rolling(10).sum()
        
        # If we have orderbook data, add real microstructure features
        if orderbook_data:
            features = self._add_orderbook_features(features, orderbook_data)
            
        return features
    
    def compute_derivatives_features(self, spot_df: pd.DataFrame, funding_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compute derivatives-based features for crypto analysis.
        """
        features = spot_df.copy()
        
        # Funding rate impact (if available)
        if funding_data:
            # Add funding rate features
            features['funding_rate'] = funding_data.get('funding_rate', 0)
            features['funding_zscore'] = self._zscore(pd.Series(funding_data.get('funding_history', [0] * len(features))), 30)
        else:
            # Estimate funding pressure from price momentum
            features['estimated_funding_pressure'] = features['Close'].pct_change(8).rolling(24).mean() * 1000
            features['funding_zscore'] = self._zscore(features['estimated_funding_pressure'], 30)
        
        # Basis estimation (spot vs perpetual proxy)
        features['basis_proxy'] = features['Close'].pct_change() * 100
        features['basis_momentum'] = features['basis_proxy'].rolling(24).mean()
        
        # Liquidation burst detection (volume spikes + price gaps)
        features['gap_size'] = abs(features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
        features['volume_spike'] = features['Volume'] / features['Volume'].rolling(50).mean()
        features['liquidation_risk'] = features['gap_size'] * features['volume_spike']
        features['liquidation_zscore'] = self._zscore(features['liquidation_risk'], 20)
        
        return features
    
    def compute_cross_asset_features(self, crypto_df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute cross-asset beta and correlation features.
        """
        features = crypto_df.copy()
        
        if btc_df is not None and len(btc_df) >= len(crypto_df):
            # Align dataframes
            aligned_btc = btc_df.tail(len(crypto_df)).reset_index(drop=True)
            
            # Compute returns
            crypto_returns = features['Close'].pct_change()
            btc_returns = aligned_btc['Close'].pct_change()
            
            # Rolling beta to BTC
            features['btc_beta'] = self._rolling_beta(crypto_returns, btc_returns, 30)
            features['btc_correlation'] = crypto_returns.rolling(30).corr(btc_returns)
            
            # Relative strength
            features['btc_relative_strength'] = crypto_returns.rolling(10).sum() - btc_returns.rolling(10).sum()
            features['btc_outperformance'] = (features['btc_relative_strength'] > 0).astype(int)
            
        else:
            # Default values when BTC data not available
            features['btc_beta'] = 1.0
            features['btc_correlation'] = 0.7
            features['btc_relative_strength'] = 0.0
            features['btc_outperformance'] = 0
        
        # Market dominance proxy (simplified)
        features['market_cap_rank'] = 10  # Default mid-cap
        features['dominance_factor'] = 1.0 / features['market_cap_rank']
        
        return features
    
    def compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market regime classification features.
        """
        features = df.copy()
        
        # Volatility regimes
        features['volatility_20d'] = features['Close'].pct_change().rolling(20).std() * np.sqrt(365) * 100
        features['vol_regime'] = pd.cut(features['volatility_20d'], 
                                      bins=[0, 30, 60, 100, np.inf], 
                                      labels=['low', 'medium', 'high', 'extreme'])
        
        # Trend regimes
        features['trend_strength'] = abs(features['Close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        features['trend_regime'] = pd.cut(features['trend_strength'], 
                                        bins=[0, 0.01, 0.05, np.inf], 
                                        labels=['sideways', 'trending', 'strong_trend'])
        
        # Mean reversion vs momentum regimes
        features['hurst_exponent'] = features['Close'].rolling(50).apply(self._hurst_exponent)
        features['momentum_regime'] = (features['hurst_exponent'] > 0.5).astype(int)
        
        # Market stress indicators
        features['drawdown'] = (features['Close'] / features['Close'].rolling(50).max() - 1) * 100
        features['stress_regime'] = (features['drawdown'] < -20).astype(int)
        
        return features
    
    def compute_multi_timeframe_features(self, dfs_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute multi-timeframe confluence features.
        """
        # Use the primary timeframe as base
        base_tf = list(dfs_by_tf.keys())[0]
        features = dfs_by_tf[base_tf].copy()
        
        # Trend alignment across timeframes
        trend_alignment = 0
        for tf_name, tf_df in dfs_by_tf.items():
            if len(tf_df) > 20:
                ema_fast = tf_df['Close'].ewm(span=10).mean()
                ema_slow = tf_df['Close'].ewm(span=20).mean()
                trend_up = (ema_fast.iloc[-1] > ema_slow.iloc[-1]).astype(int)
                trend_alignment += trend_up
        
        features['trend_alignment'] = trend_alignment / len(dfs_by_tf)
        
        # Volatility structure across timeframes
        vol_structure = []
        for tf_name, tf_df in dfs_by_tf.items():
            if len(tf_df) > 10:
                vol = tf_df['Close'].pct_change().rolling(10).std()
                vol_structure.append(vol.iloc[-1] if not pd.isna(vol.iloc[-1]) else 0)
        
        features['vol_structure_score'] = np.mean(vol_structure) if vol_structure else 0
        
        return features
    
    def create_comprehensive_features(self, primary_df: pd.DataFrame, 
                                    dfs_by_tf: Optional[Dict] = None,
                                    btc_df: Optional[pd.DataFrame] = None,
                                    orderbook_data: Optional[Dict] = None,
                                    funding_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models.
        """
        # Start with primary dataframe
        features = primary_df.copy()
        
        # Add microstructure features
        features = self.compute_microstructure_features(features, orderbook_data)
        
        # Add derivatives features
        features = self.compute_derivatives_features(features, funding_data)
        
        # Add cross-asset features
        features = self.compute_cross_asset_features(features, btc_df)
        
        # Add regime features
        features = self.compute_regime_features(features)
        
        # Add multi-timeframe features if available
        if dfs_by_tf:
            mtf_features = self.compute_multi_timeframe_features(dfs_by_tf)
            # Merge relevant columns
            for col in ['trend_alignment', 'vol_structure_score']:
                if col in mtf_features.columns:
                    features[col] = mtf_features[col]
        
        # Technical indicators (existing ones)
        features = self._add_technical_indicators(features)
        
        # Clean and validate features
        features = self._clean_features(features)
        
        return features
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standard technical indicators."""
        df = df.copy()
        
        # EMAs
        for period in [8, 20, 50, 200]:
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        df['rsi'] = self._rsi(df['Close'], 14)
        
        # ATR
        df['atr'] = self._atr(df, 14)
        df['atr_pct'] = 100 * df['atr'] / df['Close']
        
        # Bollinger Bands
        bb_mean = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = bb_mean + 2 * bb_std
        df['bb_lower'] = bb_mean - 2 * bb_std
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    # Helper methods
    def _zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling z-score."""
        return (series - series.rolling(window).mean()) / series.rolling(window).std()
    
    def _rolling_beta(self, y: pd.Series, x: pd.Series, window: int) -> pd.Series:
        """Calculate rolling beta."""
        def beta(y_window, x_window):
            covariance = np.cov(y_window, x_window)[0][1]
            variance = np.var(x_window)
            return covariance / variance if variance > 0 else 0
        
        betas = []
        for i in range(len(y)):
            if i < window:
                betas.append(np.nan)
            else:
                y_window = y.iloc[i-window:i].values
                x_window = x.iloc[i-window:i].values
                betas.append(beta(y_window, x_window))
        
        return pd.Series(betas, index=y.index)
    
    def _hurst_exponent(self, price_series: pd.Series) -> float:
        """Calculate Hurst exponent for regime detection."""
        if len(price_series) < 20:
            return 0.5
        
        try:
            lags = range(2, 20)
            tau = []
            
            for lag in lags:
                price_arr = np.array(price_series)
                pp = price_arr[lag:] - price_arr[:-lag]
                tau.append(np.sqrt(np.std(pp)))
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def _rsi(self, price: pd.Series, window: int) -> pd.Series:
        """Calculate RSI."""
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(tr).rolling(window).mean()
    
    def _add_orderbook_features(self, features: pd.DataFrame, orderbook_data: Dict) -> pd.DataFrame:
        """Add real orderbook features if available."""
        # This would be implemented with real orderbook data
        features['bid_ask_spread'] = orderbook_data.get('spread', 0.001)
        features['orderbook_imbalance'] = orderbook_data.get('imbalance', 0.0)
        features['market_depth'] = orderbook_data.get('depth', 1000000)
        return features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate feature dataframe."""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill small gaps
        features = features.ffill(limit=5)
        
        # Drop remaining NaN rows
        features = features.dropna()
        
        return features

# Global instance for use across the application
feature_engine = AdvancedFeatureEngine()
