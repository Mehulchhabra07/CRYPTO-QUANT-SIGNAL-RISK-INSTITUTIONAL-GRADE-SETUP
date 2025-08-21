"""
Advanced Risk Management System
Implements CVaR, correlation clustering, fractional Kelly sizing, and circuit breakers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    """
    Hedge-fund grade risk management system.
    """
    
    def __init__(self, max_portfolio_risk: float = 0.02, 
                 max_single_position: float = 0.05,
                 confidence_level: float = 0.05):
        self.max_portfolio_risk = max_portfolio_risk  # 2% portfolio VaR
        self.max_single_position = max_single_position  # 5% max single position
        self.confidence_level = confidence_level  # 95% confidence level
        
        # Circuit breakers
        self.daily_loss_limit = 0.01  # 1% daily loss limit
        self.weekly_dd_limit = 0.05   # 5% weekly drawdown limit
        self.consecutive_loss_limit = 3
        
        # State tracking
        self.daily_pnl = 0.0
        self.weekly_dd = 0.0
        self.consecutive_losses = 0
        self.position_history = []
        
    def fractional_kelly_sizing(self, expected_return: float,
                               prob_success: float,
                               prob_uncertainty: float = 0.1,
                               win_loss_ratio: float = 1.5,
                               kelly_fraction: float = 0.25) -> float:
        """
        Calculate position size using Fractional Kelly with uncertainty shrinkage.
        
        Args:
            expected_return: Expected return of the trade
            prob_success: Probability of success
            prob_uncertainty: Uncertainty in probability estimate
            win_loss_ratio: Average win / average loss ratio
            kelly_fraction: Fraction of Kelly to use (for safety)
            
        Returns:
            Position size as fraction of portfolio
        """
        
        # Shrink probability due to uncertainty
        shrunk_prob = prob_success * (1 - prob_uncertainty)
        
        # Kelly formula: f = (bp - q) / b
        # where b = win/loss ratio, p = prob of win, q = prob of loss
        if shrunk_prob <= 0 or win_loss_ratio <= 0:
            return 0.0
            
        kelly_optimal = (win_loss_ratio * shrunk_prob - (1 - shrunk_prob)) / win_loss_ratio
        kelly_optimal = max(0, kelly_optimal)  # No negative sizing
        
        # Apply fractional Kelly for safety
        position_size = kelly_optimal * kelly_fraction
        
        # Apply maximum position limit
        position_size = min(position_size, self.max_single_position)
        
        return position_size
    
    def cvar_portfolio_risk(self, positions: Dict[str, float],
                           returns_history: pd.DataFrame,
                           confidence_level: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate portfolio CVaR (Conditional Value at Risk).
        
        Args:
            positions: Dict of {symbol: position_size}
            returns_history: Historical returns for each symbol
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            (VaR, CVaR) at specified confidence level
        """
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Align positions with returns data
        symbols = list(positions.keys())
        portfolio_returns = []
        
        for i in range(len(returns_history)):
            portfolio_return = 0
            for symbol in symbols:
                if symbol in returns_history.columns:
                    portfolio_return += positions[symbol] * returns_history[symbol].iloc[i]
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, confidence_level * 100)
        
        # Calculate CVaR (expected loss beyond VaR)
        cvar = np.mean(portfolio_returns[portfolio_returns <= var])
        
        return float(var), float(cvar)
    
    def correlation_clustering(self, returns_data: pd.DataFrame,
                              n_clusters: int = 3) -> Dict[str, int]:
        """
        Cluster assets by correlation to avoid concentration risk.
        
        Args:
            returns_data: Historical returns for all assets
            n_clusters: Number of correlation clusters
            
        Returns:
            Dict mapping symbols to cluster IDs
        """
        
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        # Convert to distance matrix (1 - correlation)
        distance_matrix = 1 - corr_matrix.abs()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        # Map symbols to clusters
        symbol_clusters = {}
        for i, symbol in enumerate(returns_data.columns):
            symbol_clusters[symbol] = cluster_labels[i]
        
        return symbol_clusters
    
    def cluster_concentration_limits(self, positions: Dict[str, float],
                                   symbol_clusters: Dict[str, int],
                                   max_cluster_exposure: float = 0.3) -> Dict[str, float]:
        """
        Apply concentration limits by correlation cluster.
        
        Args:
            positions: Original position sizes
            symbol_clusters: Cluster assignments
            max_cluster_exposure: Maximum exposure per cluster
            
        Returns:
            Adjusted position sizes
        """
        
        # Calculate current cluster exposures
        cluster_exposures = {}
        for symbol, position in positions.items():
            cluster = symbol_clusters.get(symbol, 0)
            cluster_exposures[cluster] = cluster_exposures.get(cluster, 0) + abs(position)
        
        # Scale down positions if cluster limits exceeded
        adjusted_positions = positions.copy()
        
        for cluster, exposure in cluster_exposures.items():
            if exposure > max_cluster_exposure:
                scale_factor = max_cluster_exposure / exposure
                
                # Scale down all positions in this cluster
                for symbol, position in positions.items():
                    if symbol_clusters.get(symbol, 0) == cluster:
                        adjusted_positions[symbol] = position * scale_factor
        
        return adjusted_positions
    
    def check_circuit_breakers(self, current_pnl: float,
                              portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if any circuit breakers should halt trading.
        
        Returns:
            (should_halt, reason)
        """
        
        # Update daily PnL
        self.daily_pnl = current_pnl
        
        # Daily loss limit
        daily_loss_pct = self.daily_pnl / portfolio_value
        if daily_loss_pct < -self.daily_loss_limit:
            return True, f"Daily loss limit exceeded: {daily_loss_pct:.2%}"
        
        # Weekly drawdown limit
        if self.weekly_dd < -self.weekly_dd_limit:
            return True, f"Weekly drawdown limit exceeded: {self.weekly_dd:.2%}"
        
        # Consecutive losses limit
        if self.consecutive_losses >= self.consecutive_loss_limit:
            return True, f"Consecutive loss limit exceeded: {self.consecutive_losses} losses"
        
        return False, ""
    
    def update_loss_tracking(self, trade_pnl: float):
        """Update consecutive loss tracking."""
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def position_sizing_with_constraints(self, signals: Dict[str, Dict],
                                       returns_history: pd.DataFrame,
                                       portfolio_value: float) -> Dict[str, float]:
        """
        Calculate optimal position sizes considering all risk constraints.
        
        Args:
            signals: Dict of {symbol: {prob, expected_return, confidence, etc.}}
            returns_history: Historical returns data
            portfolio_value: Current portfolio value
            
        Returns:
            Dict of {symbol: position_size}
        """
        
        # Step 1: Calculate base Kelly sizes
        base_positions = {}
        for symbol, signal in signals.items():
            prob_success = signal.get('prob', 0.5)
            expected_return = signal.get('expected_return', 0.0)
            confidence = signal.get('confidence', 0.5)
            
            # Estimate uncertainty from confidence
            prob_uncertainty = 1 - confidence
            
            # Estimate win/loss ratio from expected return and probability
            if prob_success > 0 and prob_success < 1:
                win_loss_ratio = abs(expected_return / (prob_success - 1)) if prob_success != 1 else 2.0
            else:
                win_loss_ratio = 2.0
            
            position_size = self.fractional_kelly_sizing(
                expected_return, prob_success, prob_uncertainty, win_loss_ratio
            )
            
            if position_size > 0:
                base_positions[symbol] = position_size
        
        if not base_positions:
            return {}
        
        # Step 2: Apply correlation clustering constraints
        symbol_clusters = self.correlation_clustering(returns_history)
        adjusted_positions = self.cluster_concentration_limits(base_positions, symbol_clusters)
        
        # Step 3: Check portfolio-level CVaR constraint
        var, cvar = self.cvar_portfolio_risk(adjusted_positions, returns_history)
        
        if abs(cvar) > self.max_portfolio_risk:
            # Scale down all positions proportionally
            scale_factor = self.max_portfolio_risk / abs(cvar)
            for symbol in adjusted_positions:
                adjusted_positions[symbol] *= scale_factor
        
        # Step 4: Convert to dollar amounts
        dollar_positions = {}
        for symbol, fraction in adjusted_positions.items():
            dollar_positions[symbol] = fraction * portfolio_value
        
        return dollar_positions

class RegimeGating:
    """
    Market regime detection and strategy gating.
    """
    
    def __init__(self):
        self.regime_cache = {}
        
    def detect_market_regime(self, price_data: pd.DataFrame,
                           volume_data: Optional[pd.Series] = None) -> pd.Series:
        """
        Detect market regime: Trend/Mean-Revert/High-Vol/Chop.
        
        Returns:
            Series with regime labels: 0=Trend, 1=Mean-Revert, 2=High-Vol, 3=Chop
        """
        
        returns = price_data['Close'].pct_change()
        
        # Volatility regime
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        high_vol = vol_20d > vol_20d.rolling(60).quantile(0.8)
        
        # Trend strength
        trend_strength = abs(returns.rolling(20).mean() / returns.rolling(20).std())
        strong_trend = trend_strength > trend_strength.rolling(60).quantile(0.7)
        
        # Mean reversion tendency (using Hurst exponent proxy)
        mean_revert = self._calculate_mean_reversion_signal(returns)
        
        # Regime classification
        regimes = []
        for i in range(len(returns)):
            if high_vol.iloc[i]:
                regimes.append(2)  # High volatility
            elif strong_trend.iloc[i]:
                regimes.append(0)  # Trending
            elif mean_revert.iloc[i]:
                regimes.append(1)  # Mean reverting
            else:
                regimes.append(3)  # Choppy/sideways
        
        return pd.Series(regimes, index=returns.index)
    
    def _calculate_mean_reversion_signal(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate mean reversion signal using autocorrelation."""
        
        def autocorr_1(x):
            if len(x) < 2:
                return 0
            return np.corrcoef(x[:-1], x[1:])[0, 1] if len(np.unique(x)) > 1 else 0
        
        autocorr = returns.rolling(window).apply(autocorr_1)
        return autocorr < -0.1  # Negative autocorrelation indicates mean reversion
    
    def gate_strategies_by_regime(self, signals: Dict[str, Dict],
                                 current_regime: int,
                                 regime_config: Dict[int, Dict]) -> Dict[str, Dict]:
        """
        Filter and adjust signals based on current market regime.
        
        Args:
            signals: Original strategy signals
            current_regime: Current regime (0-3)
            regime_config: Configuration for each regime
            
        Returns:
            Filtered and adjusted signals
        """
        
        config = regime_config.get(current_regime, {})
        enabled_strategies = config.get('enabled_strategies', ['all'])
        signal_multiplier = config.get('signal_multiplier', 1.0)
        
        filtered_signals = {}
        
        for symbol, signal in signals.items():
            strategy_type = signal.get('strategy', 'unknown')
            
            # Check if strategy is enabled for this regime
            if 'all' in enabled_strategies or strategy_type in enabled_strategies:
                adjusted_signal = signal.copy()
                
                # Apply regime-specific multiplier
                adjusted_signal['prob'] = min(1.0, signal['prob'] * signal_multiplier)
                adjusted_signal['expected_return'] = signal['expected_return'] * signal_multiplier
                
                filtered_signals[symbol] = adjusted_signal
        
        return filtered_signals

# Global instances
risk_manager = AdvancedRiskManager()
regime_gating = RegimeGating()
