"""
🚀 ULTIMATE HYBRID SCANNER 🚀
Combines ALL rule-based str        # Load ALL ML models from all ensemble files
        model_files = [
            "models/diverse_ensemble.pkl",
            "models/hedge_fund_ensemble.pkl", 
            "models/enhanced_ensemble.pkl",
            "models/beast_mode_ensemble.pkl"
        ]
        
        # E) Feature noise reduction - track noisy features
        self.noisy_features = [
            'fractal_dimension', 'hurst_exponent', 'entropy_measure', 
            'chaos_indicator', 'complexity_index'
        ]
        
        total_models = 0ith ALL 56 ML models
Scans 200 cryptos with comprehensive analysis
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import sys
sys.path.append('.')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Import all our components
from app.providers.crypto import CryptoDataManager
from app.signals import (
    multi_tf_indicators, passes_risk_gates, score_asset_tf,
    strategy_trend_swing, strategy_donchian_breakout, strategy_squeeze
)
from ml.advanced_features import feature_engine
from ml.advanced_labeling import advanced_labeling, advanced_metrics
from ml.advanced_risk import AdvancedRiskManager, RegimeGating
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class UltimateHybridScanner:
    """
    Ultimate Hybrid Scanner combining ALL strategies and models:
    - All rule-based strategies (trend swing, donchian, squeeze)
    - All 56 ML models from 4 ensembles
    - Advanced feature engineering (43+ features)
    - Regime detection and risk management
    - 200 crypto comprehensive scanning
    """
    
    def __init__(self, config_path: str = "app/config.yaml"):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize components
        self.crypto_manager = CryptoDataManager()
        self.regime_detector = RegimeGating()
        
        # Initialize advanced feature engine and risk manager
        from ml.advanced_features import AdvancedFeatureEngine
        self.advanced_features = AdvancedFeatureEngine()
        self.advanced_risk = AdvancedRiskManager()
        
        # E) Feature noise reduction - track noisy features
        self.noisy_features = [
            'fractal_dimension', 'hurst_exponent', 'entropy_measure', 
            'chaos_indicator', 'complexity_index'
        ]
        
        # Load ALL ML models from all ensembles (removing duplicates)
        self.models = []
        self.model_weights = []
        self.model_columns = None
        self.feature_columns = None
        self.load_all_ml_models()
        
        self.logger.info(f"Ultimate Hybrid Scanner initialized with {len(self.models)} unique ML models")
    
    def load_all_ml_models(self):
        """Load all ML models from all ensemble files, removing duplicates."""
        model_files = [
            "models/diverse_ensemble.pkl",
            "models/hedge_fund_ensemble.pkl", 
            "models/beast_mode_ensemble.pkl"
        ]
        
        # Track unique models by name to detect duplicates
        unique_models = {}
        model_sources = {}
        total_models_loaded = 0
        duplicates_found = 0
        
        for model_file in model_files:
            if Path(model_file).exists():
                try:
                    ensemble_data = joblib.load(model_file)
                    
                    if isinstance(ensemble_data, dict) and 'models' in ensemble_data:
                        ensemble_models = ensemble_data['models']
                        ensemble_weights = ensemble_data.get('weights', {})
                        file_models_loaded = 0
                        file_duplicates_skipped = 0
                        
                        for model_name, model_obj in ensemble_models.items():
                            total_models_loaded += 1
                            
                            if model_name in unique_models:
                                # Duplicate found - skip it
                                duplicates_found += 1
                                file_duplicates_skipped += 1
                                self.logger.debug(f"Skipping duplicate model '{model_name}' from {model_file} "
                                                f"(original from {model_sources[model_name]})")
                                continue
                            
                            # Add unique model
                            unique_models[model_name] = model_obj
                            model_sources[model_name] = model_file
                            self.models.append(model_obj)
                            
                            weight = ensemble_weights.get(model_name, 1.0)
                            self.model_weights.append(weight)
                            file_models_loaded += 1
                        
                        # Use the most comprehensive column set (usually from beast mode)
                        if 'columns' in ensemble_data:
                            if self.model_columns is None or len(ensemble_data['columns']) > len(self.model_columns):
                                self.model_columns = ensemble_data['columns']
                        
                        self.logger.info(f"Loaded {file_models_loaded} unique models from {model_file} "
                                       f"({file_duplicates_skipped} duplicates skipped)")
                
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_file}: {e}")
        
        self.logger.info(f"Total models processed: {total_models_loaded}")
        self.logger.info(f"Unique models loaded: {len(self.models)}")
        self.logger.info(f"Duplicates removed: {duplicates_found}")
        
        # Log duplicate details
        if duplicates_found > 0:
            self.logger.info("Duplicate models found and removed:")
            duplicate_names = []
            model_counts = {}
            
            # Re-scan to identify which models were duplicated
            for model_file in model_files:
                if Path(model_file).exists():
                    try:
                        ensemble_data = joblib.load(model_file)
                        if isinstance(ensemble_data, dict) and 'models' in ensemble_data:
                            for model_name in ensemble_data['models'].keys():
                                model_counts[model_name] = model_counts.get(model_name, 0) + 1
                    except:
                        continue
            
            for name, count in model_counts.items():
                if count > 1:
                    self.logger.info(f"  - '{name}' appeared {count} times")
        
        # Normalize weights
        if self.model_weights:
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]
        
        # Set feature columns - this is critical for ML predictions
        self.feature_columns = self.model_columns if self.model_columns else []
        if self.feature_columns:
            self.logger.info(f"Feature columns set: {len(self.feature_columns)} features")
        else:
            self.logger.warning("No feature columns found - ML predictions may fail!")
    
    def create_comprehensive_features(self, symbol: str, dfs_by_tf: Dict) -> pd.DataFrame:
        """Create comprehensive features combining all advanced techniques."""
        try:
            primary_df = dfs_by_tf["lower"]
            
            if len(primary_df) < 50:
                return pd.DataFrame()
            
            # Get BTC data for cross-asset features - FIXED: Use same timeframe as primary data
            btc_df = None
            try:
                # Use the same lower timeframe as the primary data for consistency
                if "lower" in dfs_by_tf:
                    # Extract timeframe from primary data frequency
                    timeframe_minutes = (primary_df.index[1] - primary_df.index[0]).total_seconds() / 60
                    if timeframe_minutes <= 60:  # 1h or less
                        btc_timeframe = "1h"
                    elif timeframe_minutes <= 240:  # 4h or less
                        btc_timeframe = "4h"
                    else:
                        btc_timeframe = "1d"
                    
                    btc_data = self.crypto_manager.fetch_multi_timeframe("BTC/USDT", btc_timeframe, btc_timeframe, 200, 200)
                    if btc_data and btc_timeframe in btc_data:
                        btc_df = btc_data[btc_timeframe]
                        self.logger.debug(f"BTC data for {symbol}: {btc_df.shape} at {btc_timeframe}")
            except Exception as btc_error:
                self.logger.warning(f"BTC data fetch failed for {symbol}: {btc_error}")
                btc_df = None  # Continue without BTC data
            
            # Create comprehensive features using the advanced feature engine
            features = self.advanced_features.create_comprehensive_features(
                primary_df=primary_df,
                dfs_by_tf=dfs_by_tf,
                btc_df=btc_df,
                orderbook_data=None,
                funding_data=None
            )
            
            # CRITICAL FIX: If BTC data causes empty features, retry without BTC data
            if features.empty and btc_df is not None:
                self.logger.warning(f"BTC data caused empty features for {symbol}, retrying without BTC data")
                features = self.advanced_features.create_comprehensive_features(
                    primary_df=primary_df,
                    dfs_by_tf=dfs_by_tf,
                    btc_df=None,  # No BTC data
                    orderbook_data=None,
                    funding_data=None
                )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature creation failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def extract_features(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """Extract features from 1h and 4h dataframes - wrapper for compatibility."""
        dfs_by_tf = {
            "lower": df_1h,
            "higher": df_4h
        }
        return self.create_comprehensive_features("TEST_SYMBOL", dfs_by_tf)
    
    def ultimate_ml_score(self, features_df: pd.DataFrame) -> Tuple[float, Dict]:
        """Get prediction from ALL ML models with comprehensive scoring."""
        try:
            if features_df.empty or len(self.models) == 0:
                return 0.5, {"models_used": 0, "confidence": 0.0}
            
            # Use the latest row for prediction
            features_row = features_df.iloc[-1:].copy()
            
            # CRITICAL FIX: Handle feature mismatch properly
            ml_scores = []
            confidences = []
            models_used = 0
            
            # Load each ensemble separately with their own feature columns
            ensemble_files = [
                ("models/diverse_ensemble.pkl", "diverse"),
                ("models/hedge_fund_ensemble.pkl", "hedge_fund"), 
                ("models/beast_mode_ensemble.pkl", "beast_mode")
            ]
            
            for ensemble_file, ensemble_name in ensemble_files:
                if not Path(ensemble_file).exists():
                    continue
                    
                try:
                    ensemble_data = joblib.load(ensemble_file)
                    ensemble_models = ensemble_data.get('models', {})
                    ensemble_columns = ensemble_data.get('columns', [])
                    ensemble_weights = ensemble_data.get('weights', {})
                    
                    if not ensemble_columns:
                        self.logger.warning(f"No columns found for {ensemble_name}, skipping")
                        continue
                    
                    # Align features for this specific ensemble
                    available_features = [col for col in ensemble_columns if col in features_row.columns]
                    
                    if len(available_features) < len(ensemble_columns) * 0.5:  # Need at least 50% feature match
                        self.logger.warning(f"Poor feature alignment for {ensemble_name}: {len(available_features)}/{len(ensemble_columns)}")
                        continue
                    
                    # Create feature matrix for this ensemble
                    ensemble_features = features_row[available_features].copy()
                    
                    # Add missing features as zeros
                    for col in ensemble_columns:
                        if col not in ensemble_features.columns:
                            ensemble_features[col] = 0.0
                    
                    # Reorder to match training
                    ensemble_features = ensemble_features[ensemble_columns]
                    
                    # Get predictions from this ensemble
                    for model_name, model_obj in ensemble_models.items():
                        try:
                            # Handle different model types
                            if isinstance(model_obj, tuple):  # Scaled models
                                model, scaler = model_obj
                                features_scaled = scaler.transform(ensemble_features)
                                
                                if hasattr(model, 'predict_proba'):
                                    proba = model.predict_proba(features_scaled)[0]
                                    score = proba[1] if len(proba) > 1 else proba[0]
                                    confidence = abs(proba[1] - proba[0]) if len(proba) > 1 else 0.5
                                else:
                                    score = float(model.predict(features_scaled)[0])
                                    confidence = 0.5
                            else:  # Regular models
                                if hasattr(model_obj, 'predict_proba'):
                                    proba = model_obj.predict_proba(ensemble_features)[0]
                                    score = proba[1] if len(proba) > 1 else proba[0]
                                    confidence = abs(proba[1] - proba[0]) if len(proba) > 1 else 0.5
                                else:
                                    score = float(model_obj.predict(ensemble_features)[0])
                                    confidence = 0.5
                            
                            # Apply model weight
                            weight = ensemble_weights.get(model_name, 1.0)
                            ml_scores.append(score * weight)
                            confidences.append(confidence)
                            models_used += 1
                            
                        except Exception as model_error:
                            self.logger.debug(f"Model {model_name} in {ensemble_name} failed: {model_error}")
                            continue
                    
                except Exception as ensemble_error:
                    self.logger.warning(f"Ensemble {ensemble_name} failed: {ensemble_error}")
                    continue
            
            if not ml_scores:
                self.logger.warning("No models provided predictions")
                return 0.5, {"models_used": 0, "confidence": 0.0}
            
            # Weighted ensemble prediction
            final_score = np.mean(ml_scores) if ml_scores else 0.5
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Ensure score is in [0, 1] range
            final_score = max(0.0, min(1.0, final_score))
            
            metadata = {
                "models_used": models_used,
                "confidence": avg_confidence,
                "individual_scores": ml_scores[:10],  # Store first 10 for debugging
                "ensemble_score": final_score,
                "total_features": len(features_row.columns) if not features_row.empty else 0,
                "total_features_available": len(features_row.columns) if not features_row.empty else 0,
                "features_used_by_ensembles": sum([
                    len(ensemble_data.get('columns', [])) 
                    for ensemble_file, _ in ensemble_files 
                    if Path(ensemble_file).exists() 
                    and (ensemble_data := joblib.load(ensemble_file))
                ])
            }
            
            return final_score, metadata
            
        except Exception as e:
            self.logger.warning(f"Ultimate ML scoring failed: {e}")
            return 0.5, {"models_used": 0, "confidence": 0.0, "error": str(e)}
    
    def enhanced_rules_scoring(self, dfs_by_tf: Dict) -> Tuple[float, Dict]:
        """Enhanced rule-based strategies with improved logic."""
        try:
            lo = dfs_by_tf["lower"].dropna()
            hi = dfs_by_tf["higher"].dropna()
            
            if lo.empty or hi.empty:
                return 0.0, {"error": "no_data"}
            
            last_lo = lo.iloc[-1]
            last_hi = hi.iloc[-1]
            
            # B) Enhanced Strategy 1: Trend Swing
            # 30m: RSI > 42, 4h: RSI > 45
            lo_rsi_ok = last_lo["RSI"] > 42
            hi_rsi_ok = last_hi["RSI"] > 45
            
            # Allow EMA20>EMA100 on either TF if other has positive slope
            lo_ema_ok = last_lo["EMA_fast"] > last_lo["EMA_slow"]
            hi_ema_ok = last_hi["EMA_fast"] > last_hi["EMA_slow"]
            
            # Calculate slopes (simple diff over 3 periods)
            lo_ema_slope = 0
            hi_ema_slope = 0
            if len(lo) >= 3:
                lo_ema_slope = (last_lo["EMA_fast"] - lo["EMA_fast"].iloc[-3]) / 3
            if len(hi) >= 3:
                hi_ema_slope = (last_hi["EMA_fast"] - hi["EMA_fast"].iloc[-3]) / 3
            
            # Early trend adoption: EMA ok on either TF if other has positive slope
            trend_condition = (lo_ema_ok and hi_rsi_ok) or (hi_ema_ok and lo_rsi_ok)
            if not trend_condition:
                trend_condition = (lo_ema_ok and hi_ema_slope > 0) or (hi_ema_ok and lo_ema_slope > 0)
            
            trend_score = 1.0 if (trend_condition and lo_rsi_ok and hi_rsi_ok) else 0.0
            
            # Enhanced Strategy 2: Donchian Breakout
            breakout_signal = last_lo["Close"] > last_lo["DonchianHi20"]
            donchian_rising = lo["DonchianHi55"].diff().tail(5).mean() > 0
            
            # Enhanced Bollinger conditions
            current_z = last_lo["BOLLz"]
            z_condition = current_z <= 2.8
            
            # Calculate Bollinger Bands for bandwidth analysis
            boll_window = 20
            lo_close = lo["Close"]
            boll_mean = lo_close.rolling(boll_window).mean()
            boll_std = lo_close.rolling(boll_window).std()
            boll_upper = boll_mean + (2 * boll_std)
            boll_lower = boll_mean - (2 * boll_std)
            
            # Bandwidth expanding check
            bb_bandwidth = (boll_upper - boll_lower) / lo_close
            bandwidth_expanding = bb_bandwidth.diff().tail(3).mean() > 0
            bollinger_ok = z_condition or bandwidth_expanding
            
            # Pre-break compression: bandwidth in bottom 40th percentile
            if len(lo) >= 60:
                bandwidth_p40 = bb_bandwidth.tail(60).quantile(0.40)
                current_bandwidth = bb_bandwidth.iloc[-1]
                compression_ok = current_bandwidth <= bandwidth_p40
            else:
                compression_ok = True
            
            breakout_score = 1.0 if (breakout_signal and donchian_rising and bollinger_ok and compression_ok) else 0.0
            
            # Enhanced Strategy 3: Squeeze Strategy
            rv_20 = lo["RV_Parkinson"].rolling(20).mean().iloc[-1] if len(lo) >= 20 else 0
            rv_100 = lo["RV_Parkinson"].rolling(100).mean().iloc[-1] if len(lo) >= 100 else rv_20
            
            base_squeeze = rv_20 < rv_100
            
            # Allow entry if RV slope turns up and close > mid-band
            rv_slope = 0
            if len(lo) >= 5:
                rv_slope = lo["RV_Parkinson"].rolling(20).mean().diff().tail(5).mean()
            
            mid_band = (boll_upper.iloc[-1] + boll_lower.iloc[-1]) / 2
            above_mid = last_lo["Close"] > mid_band
            
            squeeze_condition = base_squeeze or (rv_slope > 0 and above_mid)
            ema_aligned = last_lo["EMA_fast"] > last_lo["EMA_slow"]
            
            squeeze_score = 1.0 if (squeeze_condition and ema_aligned) else 0.0
            
            # Technical scoring (unchanged)
            tech_score, tech_meta = score_asset_tf(dfs_by_tf, self.cfg)
            tech_score = max(0, tech_score)
            
            # Combine all rule-based signals
            strategy_weights = {
                'trend': 0.3,
                'breakout': 0.25,
                'squeeze': 0.2,
                'technical': 0.25
            }
            
            combined_score = (
                trend_score * strategy_weights['trend'] +
                breakout_score * strategy_weights['breakout'] +
                squeeze_score * strategy_weights['squeeze'] +
                tech_score * strategy_weights['technical']
            )
            
            metadata = {
                'trend_signal': trend_score > 0,
                'trend_msg': "enhanced_trend_swing",
                'breakout_signal': breakout_score > 0,
                'breakout_msg': "enhanced_donchian_breakout",
                'squeeze_signal': squeeze_score > 0,
                'squeeze_msg': "enhanced_squeeze",
                'technical_score': tech_score,
                'combined_score': combined_score,
                'strategies_fired': sum([trend_score > 0, breakout_score > 0, squeeze_score > 0]),
                **tech_meta
            }
            
            return combined_score, metadata
            
        except Exception as e:
            self.logger.error(f"Enhanced rules scoring failed: {e}")
            return 0.0, {"error": str(e)}
    
    def enhanced_hybrid_scoring(self, rules_score: float, rules_meta: Dict,
                              ml_score: float, ml_meta: Dict, regime: int = 0) -> Tuple[float, bool, Dict]:
        """Enhanced hybrid scoring with new thresholds and agreement boosts."""
        
        # C) Updated regime-specific configuration
        regime_config = {
            0: {"rules_weight": 0.4, "ml_weight": 0.6, "min_threshold": 0.25},  # Trending
            1: {"rules_weight": 0.5, "ml_weight": 0.5, "min_threshold": 0.30}, # Mean-revert
            2: {"rules_weight": 0.6, "ml_weight": 0.4, "min_threshold": 0.32}, # High-vol
            3: {"rules_weight": 0.45, "ml_weight": 0.55, "min_threshold": 0.34} # Choppy
        }
        
        config = regime_config.get(regime, regime_config[0])
        
        # Calculate weighted hybrid score
        hybrid_score = (
            rules_score * config["rules_weight"] +
            ml_score * config["ml_weight"]
        )
        
        # Enhanced agreement boost
        ml_confidence = ml_meta.get("confidence", 0.5)
        strategies_fired = rules_meta.get("strategies_fired", 0)
        ml_only_mode = rules_meta.get("ml_only_mode", False)
        
        # New boost criteria
        boost_applied = False
        if strategies_fired >= 2 and ml_confidence >= 0.7:
            hybrid_score *= 1.25  # +25% boost for truly high quality
            boost_applied = True
        elif strategies_fired == 1 and ml_confidence >= 0.8:
            hybrid_score *= 1.12  # +12% boost for strong ML with one strategy
            boost_applied = True
        elif strategies_fired >= 2 and ml_confidence >= 0.3:
            hybrid_score *= 1.15  # +15% boost (original strong agreement)
            boost_applied = True
        elif strategies_fired >= 1 and ml_confidence >= 0.4:
            hybrid_score *= 1.08  # +8% boost (original moderate agreement)
            boost_applied = True
        
        # Special handling for ML-only mode
        if ml_only_mode:
            hybrid_score *= 0.75  # Reduce score for ML-only (more conservative)
        
        # Apply regime-specific threshold
        is_vetoed = hybrid_score < config["min_threshold"]
        
        # Ensure score is bounded
        hybrid_score = max(0.0, min(1.0, hybrid_score))
        
        metadata = {
            "regime": regime,
            "rules_weight": config["rules_weight"],
            "ml_weight": config["ml_weight"],
            "threshold": config["min_threshold"],
            "strategies_fired": strategies_fired,
            "ml_confidence": ml_confidence,
            "agreement_boost": boost_applied,
            "hybrid_score": hybrid_score,
            "ml_only_mode": ml_only_mode,
            "veto_reason": "below_threshold" if is_vetoed else None
        }
        
        return hybrid_score, is_vetoed, metadata
    
    def advanced_risk_gates(self, dfs: Dict) -> Tuple[bool, str]:
        """Enhanced risk gates with adaptive thresholds."""
        try:
            lo = dfs["lower"].dropna()
            hi = dfs["higher"].dropna()
            
            if lo.empty or hi.empty:
                return False, "insufficient_data"
            
            last_lo = lo.iloc[-1]
            last_hi = hi.iloc[-1]
            
            # A) Enhanced ATR checks
            # 30m: 2.2% or <= p60 of last 90 bars
            if len(lo) >= 90:
                lo_atr_p60 = lo["ATR_pct"].tail(90).quantile(0.60)
                lo_atr_threshold = min(2.2, lo_atr_p60)
            else:
                lo_atr_threshold = 2.2
            
            # 4h: 1.8% or <= p65 of last 180 bars  
            if len(hi) >= 180:
                hi_atr_p65 = hi["ATR_pct"].tail(180).quantile(0.65)
                hi_atr_threshold = min(1.8, hi_atr_p65)
            else:
                hi_atr_threshold = 1.8
            
            # Z-score alternative: ATR z-score <= 1.5
            if len(lo) >= 100:
                lo_atr_zscore = (last_lo["ATR_pct"] - lo["ATR_pct"].tail(100).mean()) / (lo["ATR_pct"].tail(100).std() + 1e-9)
                lo_atr_ok = (last_lo["ATR_pct"] <= lo_atr_threshold) or (lo_atr_zscore <= 1.5)
            else:
                lo_atr_ok = last_lo["ATR_pct"] <= lo_atr_threshold
            
            if len(hi) >= 100:
                hi_atr_zscore = (last_hi["ATR_pct"] - hi["ATR_pct"].tail(100).mean()) / (hi["ATR_pct"].tail(100).std() + 1e-9)
                hi_atr_ok = (last_hi["ATR_pct"] <= hi_atr_threshold) or (hi_atr_zscore <= 1.5)
            else:
                hi_atr_ok = last_hi["ATR_pct"] <= hi_atr_threshold
            
            if not (lo_atr_ok and hi_atr_ok):
                return False, "atr_too_high"
            
            # B) Enhanced liquidity check: $notional_per_minute >= $10k
            if len(lo) >= 30:
                # Calculate notional per minute for last 30 bars
                lo_volume_usd = lo["Close"] * lo.get("Volume", 0)
                notional_per_min = lo_volume_usd.tail(30)
                min_notional_per_min = notional_per_min.min()
                
                if min_notional_per_min < 10000:  # $10k minimum
                    return False, "liquidity_too_low"
            
            return True, "passed"
            
        except Exception as e:
            return False, f"risk_gate_error: {e}"

    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """Comprehensive scan of a single symbol with all strategies."""
        try:
            # Fetch multi-timeframe data
            lower_tf = self.cfg.get("data", {}).get("crypto_lower_tf", "30m")
            higher_tf = self.cfg.get("data", {}).get("crypto_higher_tf", "4h")
            
            dfs_raw = self.crypto_manager.fetch_multi_timeframe(symbol, lower_tf, higher_tf, 500, 500)
            
            if not dfs_raw or len(dfs_raw) < 2:
                return None
            
            # Convert to expected format
            dfs = {
                "lower": dfs_raw.get(lower_tf),
                "higher": dfs_raw.get(higher_tf)
            }
            
            if dfs["lower"] is None or dfs["higher"] is None:
                return None
            
            # Add technical indicators
            dfs = multi_tf_indicators(dfs, self.cfg)
            
            # Enhanced risk gates
            passes, reason = self.advanced_risk_gates(dfs)
            if not passes:
                return None
            
            # Regime detection
            regime_data = dfs["lower"]
            regime = self.regime_detector.detect_market_regime(regime_data).iloc[-1] if len(regime_data) > 50 else 0
            
            # Enhanced rules-based scoring
            rules_score, rules_meta = self.enhanced_rules_scoring(dfs)
            
            # C) Allow ML-only if rules score is zero but ML is strong
            ml_only_allowed = False
            if rules_score <= 0:
                # Create features first to check ML capability
                features_df = self.create_comprehensive_features(symbol, dfs)
                if not features_df.empty:
                    ml_score, ml_meta = self.ultimate_ml_score(features_df)
                    ml_prob = ml_meta.get("confidence", 0.0)
                    
                    # Allow ML-only if ML_prob >= 0.72 and implied RR >= 2.0
                    if ml_prob >= 0.72:
                        # Calculate implied risk-reward (simplified)
                        current_atr = rules_meta.get("atr_pct", 1.0)
                        implied_rr = 3.0 / max(current_atr, 0.5)  # Assume 3% target vs ATR stop
                        
                        if implied_rr >= 2.0:
                            ml_only_allowed = True
                            rules_score = 0.1  # Minimal rules score for ML-only
                            rules_meta["ml_only_mode"] = True
                            rules_meta["ml_only_prob"] = ml_prob
                            rules_meta["implied_rr"] = implied_rr
            
            if rules_score <= 0 and not ml_only_allowed:
                return None
            
            # Create comprehensive features
            features_df = self.create_comprehensive_features(symbol, dfs)
            
            # E) Reduce noise from fractal features (optional weight reduction)
            if not features_df.empty:
                for noisy_feature in self.noisy_features:
                    if noisy_feature in features_df.columns:
                        # Reduce impact of noisy features by 50%
                        features_df[noisy_feature] *= 0.5
            
            # Ultimate ML scoring
            ml_score, ml_meta = self.ultimate_ml_score(features_df)
            
            # Enhanced hybrid scoring
            final_score, is_vetoed, hybrid_meta = self.enhanced_hybrid_scoring(
                rules_score, rules_meta, ml_score, ml_meta, regime
            )
            
            if is_vetoed:
                return None
            
            # Calculate trading levels for execution
            current_price = float(dfs["lower"].iloc[-1]["Close"])
            current_atr = float(rules_meta.get("atr", current_price * 0.02))
            
            # Trading levels based on REALISTIC SHORT-TERM swing trading (2-7 days)
            buy_price = current_price  # Current market price
            
            # DYNAMIC targets based on actual analysis - NOT HARDCODED
            # Calculate adaptive stop and target based on ATR, volatility, confidence
            
            # Base stop distance as multiple of ATR
            atr_multiplier = 1.5  # Conservative base multiplier
            
            # Adjust based on ML confidence and hybrid score  
            confidence = ml_meta.get("confidence", 0.5)
            if confidence >= 0.8 and final_score >= 0.7:
                # Very high confidence: Tighter stops, higher targets
                stop_multiplier = 1.0  # Tighter stop
                target_multiplier = 3.0  # Higher target
            elif confidence >= 0.6 and final_score >= 0.5:
                # Good confidence: Standard levels
                stop_multiplier = 1.5
                target_multiplier = 2.5
            elif confidence >= 0.4 and final_score >= 0.4:
                # Medium confidence: Moderate levels
                stop_multiplier = 1.8
                target_multiplier = 2.0
            else:
                # Lower confidence: Wider stops, conservative targets
                stop_multiplier = 2.2  # Wider stop
                target_multiplier = 1.6  # Conservative target
            
            # Calculate dynamic levels based on ATR
            atr_stop_distance = current_atr * stop_multiplier
            atr_target_distance = current_atr * target_multiplier
            
            # Apply adaptive bounds based on symbol price (different for different price ranges)
            if current_price > 1000:  # High price symbols like BTC
                min_stop_pct = 0.008  # 0.8% minimum
                max_stop_pct = 0.035  # 3.5% maximum
                min_target_pct = 0.012  # 1.2% minimum  
                max_target_pct = 0.05   # 5% maximum
            elif current_price > 10:   # Medium price symbols like ETH
                min_stop_pct = 0.01   # 1% minimum
                max_stop_pct = 0.04   # 4% maximum
                min_target_pct = 0.015 # 1.5% minimum
                max_target_pct = 0.06  # 6% maximum
            else:  # Low price symbols (altcoins)
                min_stop_pct = 0.015  # 1.5% minimum
                max_stop_pct = 0.05   # 5% maximum
                min_target_pct = 0.02  # 2% minimum
                max_target_pct = 0.08  # 8% maximum
            
            stop_distance = max(min_stop_pct * current_price, 
                              min(max_stop_pct * current_price, atr_stop_distance))
            target_distance = max(min_target_pct * current_price,
                                min(max_target_pct * current_price, atr_target_distance))
            
            stop_loss = current_price - stop_distance
            target_price = current_price + target_distance
            
            # Calculate position sizing
            position_data = self.calculate_position_sizing(
                final_score, ml_meta.get("confidence", 0.5), regime, 
                rules_meta.get("ml_only_mode", False)
            )
            
            # Create comprehensive result with both top-level and meta fields
            result = {
                "symbol": symbol,
                "score": final_score,
                # Top-level fields for easy access
                "ml_score": ml_score,
                "ml_confidence": ml_meta.get("confidence", 0.5),
                "rule_score": rules_score,
                "hybrid_score": final_score,
                "position_size_pct": position_data["position_size"] * 100,
                "atr_pct": float(rules_meta.get("atr_pct", 0)),
                "rsi": float(rules_meta.get("rsi", 0)),
                # TRADING LEVELS FOR EXECUTION
                "buy_price": round(buy_price, 6),
                "stop_loss": round(stop_loss, 6),
                "target_price": round(target_price, 6),
                "risk_reward_ratio": round((target_price - buy_price) / (buy_price - stop_loss), 2),
                "atr_value": round(current_atr, 6),
                # Analysis breakdown
                "analysis": {
                    "timeframe_analysis": {
                        "primary_tf": "30m",
                        "confirmation_tf": "4h", 
                        "expected_holding_period": "2-7 days",
                        "time_stops": {
                            "short_term_exit": "48 hours",
                            "medium_term_exit": "4 days"
                        }
                    },
                    "ml_analysis": {
                        "models_used": ml_meta.get("models_used", 0),
                        "confidence": ml_meta.get("confidence", 0.5),
                        "total_features": ml_meta.get("total_features", 0),
                        "individual_scores": ml_meta.get("individual_scores", [])[:10],  # First 10 only
                        "ensemble_score": ml_score
                    },
                    "rule_analysis": {
                        "strategies_fired": rules_meta.get("strategies_fired", 0),
                        "trend_signal": bool(rules_meta.get("trend_signal", False)),
                        "breakout_signal": bool(rules_meta.get("breakout_signal", False)),
                        "squeeze_signal": bool(rules_meta.get("squeeze_signal", False)),
                        "technical_score": float(rules_meta.get("technical_score", 0)),
                        "ml_only_mode": rules_meta.get("ml_only_mode", False)
                    },
                    "risk_analysis": {
                        "regime": int(regime),
                        "position_size": position_data["position_size"],
                        "shrunk_prob": position_data["shrunk_prob"],
                        "time_stops": {
                            "30m_bars": position_data["time_stop_30m"],
                            "4h_bars": position_data["time_stop_4h"]
                        }
                    },
                    "hybrid_analysis": {
                        "agreement_boost": hybrid_meta.get("agreement_boost", False),
                        "veto_reason": hybrid_meta.get("veto_reason"),
                        **hybrid_meta
                    }
                },
                # Legacy meta field for compatibility
                "meta": {
                    "atr_pct": float(rules_meta.get("atr_pct", 0)),
                    "rsi": float(rules_meta.get("rsi", 0)),
                    "rules_score": rules_score,
                    "ml_score": ml_score,
                    "ml_models_used": ml_meta.get("models_used", 0),
                    "ml_confidence": ml_meta.get("confidence", 0.5),
                    "total_features": ml_meta.get("total_features", 0),
                    "regime": int(regime),
                    "strategies_fired": rules_meta.get("strategies_fired", 0),
                    "trend_signal": bool(rules_meta.get("trend_signal", False)),
                    "breakout_signal": bool(rules_meta.get("breakout_signal", False)),
                    "squeeze_signal": bool(rules_meta.get("squeeze_signal", False)),
                    "technical_score": float(rules_meta.get("technical_score", 0)),
                    "hybrid_score": float(final_score),
                    "veto_reason": hybrid_meta.get("veto_reason"),
                    "agreement_boost": hybrid_meta.get("agreement_boost", False),
                    **hybrid_meta
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Scan failed for {symbol}: {e}")
            return None
    
    def calculate_position_sizing(self, score: float, ml_confidence: float, 
                                regime: int, ml_only_mode: bool = False) -> Dict:
        """D) Calculate position sizing using Fractional Kelly and probability shrinkage."""
        
        # Base Kelly parameters
        base_kelly_fraction = 0.25
        
        # D) Probability shrinkage: p' = 0.5 + 0.5*(p-0.5)*0.8
        raw_prob = ml_confidence
        shrunk_prob = 0.5 + 0.5 * (raw_prob - 0.5) * 0.8
        
        # Risk adjustment based on mode and regime
        size_multiplier = 1.0
        
        if ml_only_mode:
            size_multiplier = 0.5  # Half size for ML-only
        elif regime == 2:  # High volatility
            size_multiplier = 0.75  # Reduce size in high vol
        elif regime == 0 and score > 0.7:  # Strong trending signal
            size_multiplier = 1.25  # Increase size for strong trends
        
        # Final position size
        position_size = base_kelly_fraction * shrunk_prob * size_multiplier
        position_size = max(0.1, min(0.5, position_size))  # Clamp between 10-50%
        
        # Time stops based on timeframe
        time_stop_30m = 96  # 96 bars = 48 hours
        time_stop_4h = 24   # 24 bars = 4 days
        
        return {
            "position_size": position_size,
            "raw_prob": raw_prob,
            "shrunk_prob": shrunk_prob,
            "size_multiplier": size_multiplier,
            "time_stop_30m": time_stop_30m,
            "time_stop_4h": time_stop_4h,
            "partial_take": 0.33,  # 1/3 at +1.25R
            "trail_multiple": 2.5   # Trail by ATR*2.5
        }

    def scan_all_cryptos(self, limit: int = 400) -> Dict:
        """Scan all cryptos with comprehensive analysis and Top-K acceptance."""
        self.logger.info(f"Starting Enhanced Ultimate Hybrid scan of {limit} crypto symbols")
        
        # Get symbols
        symbols = self.crypto_manager.get_symbols("USDT", limit)
        
        results = []
        all_candidates = []  # Store all candidates, even those below threshold
        scanned = 0
        candidates = 0
        vetoed = 0
        
        for symbol in symbols:
            try:
                self.logger.info(f"Enhanced scanning: {symbol}")
                result = self.scan_symbol(symbol)
                
                scanned += 1
                
                if result:
                    # Add position sizing
                    ml_confidence = result['meta'].get('ml_confidence', 0.5)
                    regime = result['meta'].get('regime', 0)
                    ml_only_mode = result['meta'].get('ml_only_mode', False)
                    
                    sizing = self.calculate_position_sizing(
                        result['score'], ml_confidence, regime, ml_only_mode
                    )
                    result['meta'].update(sizing)
                    
                    all_candidates.append(result)
                    candidates += 1
                else:
                    vetoed += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to scan {symbol}: {e}")
                scanned += 1
                vetoed += 1
                continue
        
        # C) Top-K acceptance controller
        # Sort all candidates by score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Separate above and below threshold
        above_threshold = []
        below_threshold = []
        
        for candidate in all_candidates:
            regime = candidate['meta'].get('regime', 0)
            threshold_map = {0: 0.25, 1: 0.30, 2: 0.32, 3: 0.34}
            threshold = threshold_map.get(regime, 0.30)
            
            if candidate['score'] >= threshold:
                above_threshold.append(candidate)
            else:
                below_threshold.append(candidate)
        
        # Always surface Top-3 if no above-threshold candidates
        top_k = 3
        final_results = above_threshold.copy()
        
        if len(above_threshold) == 0 and len(below_threshold) > 0:
            # Add top K as "stand-down candidates"
            stand_down_candidates = below_threshold[:top_k]
            for candidate in stand_down_candidates:
                candidate['meta']['stand_down_candidate'] = True
                candidate['meta']['position_size'] *= 0.33  # 25-33% size
            final_results.extend(stand_down_candidates)
        
        # Create comprehensive summary
        summary = {
            "timestamp_utc": pd.Timestamp.now(tz='UTC').isoformat(),
            "top_crypto": final_results[:50],  # Top 50 picks
            "meta": {
                "scan_mode": "enhanced_ultimate_hybrid",
                "crypto_scanned": scanned,
                "crypto_candidates": candidates,
                "crypto_vetoed": vetoed,
                "above_threshold": len(above_threshold),
                "stand_down_candidates": len([c for c in final_results if c['meta'].get('stand_down_candidate', False)]),
                "ml_models_loaded": len(self.models),
                "ml_models_total_processed": sum([
                    len(joblib.load(f)['models']) for f in [
                        "models/diverse_ensemble.pkl", "models/hedge_fund_ensemble.pkl", 
                        "models/enhanced_ensemble.pkl", "models/beast_mode_ensemble.pkl"
                    ] if Path(f).exists()
                ]),
                "ml_duplicates_removed": sum([
                    len(joblib.load(f)['models']) for f in [
                        "models/diverse_ensemble.pkl", "models/hedge_fund_ensemble.pkl", 
                        "models/enhanced_ensemble.pkl", "models/beast_mode_ensemble.pkl"
                    ] if Path(f).exists()
                ]) - len(self.models),
                "ml_enabled": True,
                "advanced_features": True,
                "regime_gating": True,
                "risk_management": True,
                "position_sizing": True,
                "top_k_acceptance": True,
                "rules_strategies": ["enhanced_trend_swing", "enhanced_donchian_breakout", "enhanced_squeeze"],
                "ml_ensembles": ["diverse", "hedge_fund", "enhanced", "beast_mode"],
                "total_features": len(self.model_columns) if self.model_columns else 0,
                "comprehensive_analysis": True,
                "enhancements": {
                    "adaptive_risk_gates": True,
                    "enhanced_rules_logic": True,
                    "ml_only_mode": True,
                    "lowered_thresholds": True,
                    "enhanced_agreement_boost": True,
                    "fractional_kelly_sizing": True,
                    "top_k_acceptance": True
                }
            }
        }
        
        self.logger.info(f"Enhanced scan complete: {scanned} scanned, {candidates} candidates, {vetoed} vetoed")
        self.logger.info(f"Results: {len(above_threshold)} above threshold, {len(final_results) - len(above_threshold)} stand-down")
        return summary

def main():
    parser = argparse.ArgumentParser(description="Ultimate Hybrid Scanner - All Strategies + All Models")
    parser.add_argument("--symbols", type=int, default=200, help="Number of symbols to scan")
    parser.add_argument("--save", type=str, default="reports/ultimate_hybrid_scan.json", help="Output file")
    
    args = parser.parse_args()
    
    # Create scanner
    scanner = UltimateHybridScanner()
    
    # Run comprehensive scan
    results = scanner.scan_all_cryptos(args.symbols)
    
    # Save results
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Display results
    print(f"\nUltimate Hybrid Scan Results:")
    print(f"Symbols scanned: {results['meta']['crypto_scanned']}")
    print(f"Candidates found: {results['meta']['crypto_candidates']}")
    print(f"ML models used: {results['meta']['ml_models_loaded']}")
    print(f"Rules strategies: {len(results['meta']['rules_strategies'])}")
    print(f"Total features: {results['meta']['total_features']}")
    
    if results['top_crypto']:
        print(f"\nTop 10 Ultimate Picks:")
        for i, pick in enumerate(results['top_crypto'][:10], 1):
            meta = pick['meta']
            print(f"{i:2d}. {pick['symbol']:12s}: {pick['score']:.3f}")
            print(f"    Rules: {meta['rules_score']:.3f} | ML: {meta['ml_score']:.3f} | Confidence: {meta['ml_confidence']:.3f}")
            print(f"    Strategies: {meta['strategies_fired']}/3 | Models: {meta['ml_models_used']}")
            print(f"    Regime: {meta['regime']} | Features: {meta['total_features']}")
    
    print(f"\nResults saved to {args.save}")

if __name__ == "__main__":
    main()
