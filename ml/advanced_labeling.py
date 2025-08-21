"""
Advanced ML Labeling System for Hedge-Fund Grade Models
Implements triple-barrier labeling, meta-labeling, and proper evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

class AdvancedLabeling:
    """
    Advanced labeling system with triple-barrier method and meta-labeling.
    """
    
    def __init__(self):
        self.label_cache = {}
    
    def triple_barrier_labels(self, prices: pd.Series, 
                            volatility: pd.Series,
                            target_return: float = 0.02,
                            stop_loss: float = 0.01,
                            max_holding_days: int = 5) -> pd.DataFrame:
        """
        Create triple-barrier labels for each observation.
        
        Args:
            prices: Price series
            volatility: Rolling volatility series (for dynamic barriers)
            target_return: Target profit threshold (default 2%)
            stop_loss: Stop loss threshold (default 1%)
            max_holding_days: Maximum holding period
            
        Returns:
            DataFrame with labels: 1=profit target hit, -1=stop loss hit, 0=time expired
        """
        
        labels = []
        label_info = []
        
        for i in range(len(prices) - max_holding_days):
            entry_price = prices.iloc[i]
            current_vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.02
            
            # Dynamic barriers based on volatility
            profit_target = entry_price * (1 + max(target_return, current_vol * 2))
            stop_loss_level = entry_price * (1 - max(stop_loss, current_vol * 1.5))
            
            # Look forward to find which barrier is hit first
            label = 0  # Default: time barrier
            hit_day = max_holding_days
            exit_price = prices.iloc[i + max_holding_days]
            exit_return = (exit_price - entry_price) / entry_price
            
            for day in range(1, max_holding_days + 1):
                if i + day >= len(prices):
                    break
                    
                future_price = prices.iloc[i + day]
                
                # Check profit target
                if future_price >= profit_target:
                    label = 1
                    hit_day = day
                    exit_price = future_price
                    exit_return = (future_price - entry_price) / entry_price
                    break
                
                # Check stop loss
                if future_price <= stop_loss_level:
                    label = -1
                    hit_day = day
                    exit_price = future_price
                    exit_return = (future_price - entry_price) / entry_price
                    break
            
            labels.append(label)
            label_info.append({
                'label': label,
                'hit_day': hit_day,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_return': exit_return,
                'profit_target': profit_target,
                'stop_loss_level': stop_loss_level,
                'volatility': current_vol
            })
        
        # Pad the end with neutral labels
        while len(labels) < len(prices):
            labels.append(0)
            label_info.append({
                'label': 0, 'hit_day': 0, 'entry_price': np.nan,
                'exit_price': np.nan, 'exit_return': 0, 'profit_target': np.nan,
                'stop_loss_level': np.nan, 'volatility': np.nan
            })
        
        result = pd.DataFrame(label_info, index=prices.index)
        return result
    
    def meta_labeling(self, primary_predictions: pd.Series,
                     features: pd.DataFrame,
                     actual_returns: pd.Series,
                     threshold: float = 0.6) -> pd.DataFrame:
        """
        Meta-labeling to improve precision by filtering false positives.
        
        Args:
            primary_predictions: Primary model predictions (0-1)
            features: Feature matrix
            actual_returns: Actual forward returns
            threshold: Threshold for primary model confidence
            
        Returns:
            DataFrame with meta-labels (1=trade, 0=no trade)
        """
        
        # Only consider high-confidence primary predictions
        high_confidence = primary_predictions >= threshold
        
        # Create meta-labels based on whether high-confidence predictions were profitable
        meta_labels = []
        for i in range(len(primary_predictions)):
            if high_confidence.iloc[i]:
                # Was the primary prediction correct?
                if primary_predictions.iloc[i] > 0.5 and actual_returns.iloc[i] > 0:
                    meta_labels.append(1)  # Trade
                elif primary_predictions.iloc[i] <= 0.5 and actual_returns.iloc[i] <= 0:
                    meta_labels.append(1)  # Trade
                else:
                    meta_labels.append(0)  # Don't trade (false positive)
            else:
                meta_labels.append(0)  # Don't trade (low confidence)
        
        return pd.DataFrame({
            'meta_label': meta_labels,
            'primary_prediction': primary_predictions,
            'high_confidence': high_confidence,
            'actual_return': actual_returns
        }, index=primary_predictions.index)
    
    def regime_aware_labels(self, prices: pd.Series, 
                           regime_indicator: pd.Series,
                           regimes: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create regime-specific labels with different parameters per regime.
        
        Args:
            prices: Price series
            regime_indicator: Series indicating current regime (0, 1, 2, etc.)
            regimes: Dict mapping regime numbers to parameter dicts
            
        Returns:
            DataFrame with regime-aware labels
        """
        
        volatility = prices.pct_change().rolling(20).std()
        all_labels = []
        
        for regime_id, params in regimes.items():
            regime_mask = (regime_indicator == regime_id)
            
            if regime_mask.sum() > 0:
                regime_prices = prices[regime_mask]
                regime_vol = volatility[regime_mask]
                
                regime_labels = self.triple_barrier_labels(
                    regime_prices, 
                    regime_vol,
                    target_return=params.get('target_return', 0.02),
                    stop_loss=params.get('stop_loss', 0.01),
                    max_holding_days=params.get('max_holding_days', 5)
                )
                
                regime_labels['regime'] = regime_id
                all_labels.append(regime_labels)
        
        if all_labels:
            combined_labels = pd.concat(all_labels).sort_index()
            return combined_labels.reindex(prices.index, fill_value=0)
        else:
            # Fallback to basic labeling
            return self.triple_barrier_labels(prices, volatility)

class AdvancedValidation:
    """
    Advanced validation techniques for time series ML models.
    """
    
    def purged_walk_forward_split(self, X: pd.DataFrame, 
                                 y: pd.Series,
                                 n_splits: int = 5,
                                 test_size: float = 0.2,
                                 purge_days: int = 5) -> List[Tuple]:
        """
        Purged walk-forward cross-validation for time series.
        
        Args:
            X: Feature matrix
            y: Target series
            n_splits: Number of CV splits
            test_size: Fraction of data for testing
            purge_days: Days to purge between train/test to avoid lookahead
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        
        splits = []
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        
        for i in range(n_splits):
            # Calculate split boundaries
            test_start = int(n_samples * (i + 1) / n_splits) - test_samples
            test_end = int(n_samples * (i + 1) / n_splits)
            train_end = test_start - purge_days
            
            if train_end < test_samples:  # Need minimum training data
                continue
                
            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, min(test_end, n_samples)))
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def embargo_split(self, X: pd.DataFrame, 
                     y: pd.Series,
                     embargo_days: int = 10) -> Tuple:
        """
        Create train/test split with embargo period to prevent data leakage.
        """
        n_samples = len(X)
        test_start = int(n_samples * 0.8)  # Last 20% for testing
        train_end = test_start - embargo_days
        
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, n_samples))
        
        return train_idx, test_idx

class AdvancedMetrics:
    """
    Advanced evaluation metrics for trading models.
    """
    
    def __init__(self):
        pass
    
    def trading_metrics(self, y_true: np.ndarray, 
                       y_pred_proba: np.ndarray,
                       returns: np.ndarray,
                       threshold: float = 0.5) -> Dict:
        """
        Calculate comprehensive trading-specific metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            returns: Actual returns for each prediction
            threshold: Decision threshold
            
        Returns:
            Dictionary of metrics
        """
        
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Basic classification metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        
        # Trading-specific metrics
        trade_returns = returns[y_pred == 1]
        if len(trade_returns) > 0:
            avg_return_per_trade = np.mean(trade_returns)
            win_rate = np.mean(trade_returns > 0)
            profit_factor = np.sum(trade_returns[trade_returns > 0]) / abs(np.sum(trade_returns[trade_returns < 0])) if np.sum(trade_returns < 0) != 0 else np.inf
            max_loss = np.min(trade_returns) if len(trade_returns) > 0 else 0
        else:
            avg_return_per_trade = 0
            win_rate = 0
            profit_factor = 0
            max_loss = 0
        
        # Probability calibration metrics
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Expected value calculation
        expected_value = np.mean(y_pred_proba * returns)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'brier_score': brier_score,
            'avg_return_per_trade': avg_return_per_trade,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_loss': max_loss,
            'expected_value': expected_value,
            'num_trades': np.sum(y_pred == 1),
            'trade_frequency': np.sum(y_pred == 1) / len(y_pred)
        }
    
    def reliability_curve(self, y_true: np.ndarray, 
                         y_pred_proba: np.ndarray,
                         n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reliability curve for probability calibration assessment.
        
        Returns:
            (bin_centers, bin_accuracies)
        """
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                
                bin_centers.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
        
        return np.array(bin_centers), np.array(bin_accuracies)
    
    def information_coefficient(self, y_true: np.ndarray, 
                              y_pred: np.ndarray) -> float:
        """
        Calculate Information Coefficient (IC) - correlation between predictions and outcomes.
        """
        return np.corrcoef(y_true, y_pred)[0, 1] if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0

# Global instances
advanced_labeling = AdvancedLabeling()
advanced_validation = AdvancedValidation()
advanced_metrics = AdvancedMetrics()
