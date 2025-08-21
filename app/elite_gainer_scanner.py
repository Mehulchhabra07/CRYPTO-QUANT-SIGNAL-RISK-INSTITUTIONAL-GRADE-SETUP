"""
🚀 ELITE GAINER SCANNER 🚀
Top gainers with ML validation for day trading
Focus: Find the biggest movers with highest profit potential
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple, Optional
import logging
import ccxt
from dataclasses import dataclass
from enum import Enum
import time

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)

class SignalGrade(Enum):
    """Signal quality grades for day trading"""
    TOP_GAINER = "top_gainer"       # Best gainers with ML validation
    STRONG_MOVER = "strong_mover"   # Strong momentum with good volume
    MOMENTUM = "momentum"           # Basic momentum signals
    SPECULATIVE = "speculative"     # High risk/reward opportunities

@dataclass
class GainerSignal:
    """Elite gainer signal for day trading"""
    symbol: str
    grade: SignalGrade
    change_24h: float
    change_1h: float
    change_5m: float
    current_price: float
    volume_24h: float
    volume_ratio: float
    ml_score: float
    ml_confidence: float
    entry_price: float
    stop_loss: float
    target_quick: float
    target_stretch: float
    position_size_pct: float
    risk_pct: float
    reward_potential_pct: float
    risk_reward_ratio: float
    momentum_score: float
    volatility_score: float
    liquidity_score: float
    timestamp: datetime

class EliteGainerScanner:
    """
    Elite scanner focused on top gainers with ML validation
    - Scans 400+ symbols for biggest movers
    - Applies ML models for validation
    - Generates day trading signals with adaptive stops/targets
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # Load ML models if available
        self.models = []
        self.model_weights = []
        self.feature_columns = None
        self.load_ml_models()
        
        # Configuration optimized for top gainers
        self.config = {
            # Gainer thresholds
            'min_24h_change': 2.0,           # Minimum 2% 24h gain
            'min_1h_change': 0.5,            # Minimum 0.5% 1h gain
            'min_volume_24h': 100000,        # Minimum $100k volume
            'min_volume_ratio': 1.2,         # Volume ratio vs average
            
            # Day trading parameters
            'stop_loss_min': 0.8,            # Minimum 0.8% stop
            'stop_loss_max': 2.5,            # Maximum 2.5% stop
            'target_quick_min': 1.5,         # Minimum 1.5% quick target
            'target_stretch_min': 3.0,       # Minimum 3.0% stretch target
            
            # Position sizing
            'max_position_size': 15.0,       # Maximum 15% position
            'base_position_size': 5.0,       # Base 5% position
            'max_total_risk': 1.0,           # Maximum 1% total portfolio risk
            
            # Quality filters
            'min_ml_score': 0.3,             # Relaxed for gainers
            'min_ml_confidence': 0.25,       # Relaxed for gainers
            'max_symbols_to_scan': 500,      # Scan up to 500 symbols
            'max_signals_output': 50,        # Output top 50 signals
        }
        
        self.logger.info("🚀 Elite Gainer Scanner initialized")
    
    def load_ml_models(self):
        """Load ML models for signal validation"""
        try:
            import joblib
            from pathlib import Path
            
            model_files = [
                "models/diverse_ensemble.pkl",
                "models/hedge_fund_ensemble.pkl", 
                "models/beast_mode_ensemble.pkl"
            ]
            
            for model_file in model_files:
                if Path(model_file).exists():
                    try:
                        ensemble_data = joblib.load(model_file)
                        if isinstance(ensemble_data, dict) and 'models' in ensemble_data:
                            for model_name, model_obj in ensemble_data['models'].items():
                                self.models.append(model_obj)
                                self.model_weights.append(1.0)
                            
                            if 'columns' in ensemble_data:
                                self.feature_columns = ensemble_data['columns']
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.models)} ML models for validation")
            
        except Exception as e:
            self.logger.warning(f"ML models not available: {e}")
    
    def get_top_symbols(self, limit: int = 500) -> List[str]:
        """Get top trading symbols by volume"""
        try:
            self.logger.info(f"Fetching top {limit} symbols by volume...")
            
            # Get 24h tickers for all USDT pairs
            tickers = self.exchange.fetch_tickers()
            
            # Filter for USDT pairs with good volume
            usdt_tickers = []
            for symbol, ticker in tickers.items():
                if (symbol.endswith('/USDT') and 
                    ticker.get('quoteVolume', 0) > self.config['min_volume_24h']):
                    usdt_tickers.append({
                        'symbol': symbol,
                        'volume': ticker.get('quoteVolume', 0),
                        'change': ticker.get('percentage', 0),
                        'price': ticker.get('last', 0)
                    })
            
            # Sort by volume descending
            usdt_tickers.sort(key=lambda x: x['volume'], reverse=True)
            
            symbols = [t['symbol'] for t in usdt_tickers[:limit]]
            self.logger.info(f"Found {len(symbols)} USDT symbols with good volume")
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            return []
    
    def calculate_gainer_metrics(self, symbol: str) -> Optional[Dict]:
        """Calculate comprehensive gainer metrics for a symbol"""
        try:
            # Fetch multiple timeframes for analysis
            data_1d = self.exchange.fetch_ohlcv(symbol, '1d', limit=30)
            data_1h = self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
            data_5m = self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
            
            if not all([data_1d, data_1h, data_5m]):
                return None
            
            # Convert to DataFrames
            df_1d = pd.DataFrame(data_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h = pd.DataFrame(data_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_5m = pd.DataFrame(data_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Current metrics
            current_price = df_5m['close'].iloc[-1]
            
            # Calculate percentage changes
            change_24h = ((current_price - df_1d['close'].iloc[-2]) / df_1d['close'].iloc[-2]) * 100 if len(df_1d) > 1 else 0
            change_1h = ((current_price - df_1h['close'].iloc[-2]) / df_1h['close'].iloc[-2]) * 100 if len(df_1h) > 1 else 0
            change_5m = ((current_price - df_5m['close'].iloc[-2]) / df_5m['close'].iloc[-2]) * 100 if len(df_5m) > 1 else 0
            
            # Volume analysis
            volume_24h = df_1d['volume'].iloc[-1] * current_price
            avg_volume = df_1d['volume'].tail(7).mean() * df_1d['close'].tail(7).mean()
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility analysis (ATR-based)
            df_1h['tr'] = np.maximum(
                df_1h['high'] - df_1h['low'],
                np.maximum(
                    np.abs(df_1h['high'] - df_1h['close'].shift(1)),
                    np.abs(df_1h['low'] - df_1h['close'].shift(1))
                )
            )
            atr = df_1h['tr'].tail(14).mean()
            volatility_pct = (atr / current_price) * 100
            
            # Momentum analysis
            rsi_data = self.calculate_rsi(df_1h['close'], 14)
            macd_data = self.calculate_macd(df_1h['close'])
            
            momentum_score = self.calculate_momentum_score(
                change_24h, change_1h, change_5m, rsi_data, macd_data
            )
            
            # Liquidity score
            liquidity_score = min(1.0, volume_24h / 1000000)  # Normalized to $1M
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'change_24h': change_24h,
                'change_1h': change_1h,
                'change_5m': change_5m,
                'volume_24h': volume_24h,
                'volume_ratio': volume_ratio,
                'volatility_pct': volatility_pct,
                'atr': atr,
                'momentum_score': momentum_score,
                'liquidity_score': liquidity_score,
                'rsi': rsi_data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': histogram.iloc[-1]
            }
        except:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    def calculate_momentum_score(self, change_24h: float, change_1h: float, 
                                change_5m: float, rsi: float, macd: Dict) -> float:
        """Calculate comprehensive momentum score"""
        try:
            # Change momentum (40% weight)
            change_momentum = (
                min(change_24h / 10.0, 1.0) * 0.5 +  # 24h change normalized
                min(change_1h / 3.0, 1.0) * 0.3 +    # 1h change normalized
                min(change_5m / 1.0, 1.0) * 0.2      # 5m change normalized
            )
            
            # RSI momentum (30% weight)
            rsi_momentum = 0.0
            if 30 <= rsi <= 70:  # Sweet spot for momentum
                rsi_momentum = 1.0
            elif rsi > 70:  # Overbought but still momentum
                rsi_momentum = 0.7
            else:  # Oversold, potential reversal
                rsi_momentum = 0.5
            
            # MACD momentum (30% weight)
            macd_momentum = 0.5  # Default neutral
            if macd['histogram'] > 0:  # Positive histogram
                macd_momentum = 0.8
            if macd['macd'] > macd['signal']:  # MACD above signal
                macd_momentum += 0.2
            
            # Combined momentum score
            momentum_score = (
                change_momentum * 0.4 +
                rsi_momentum * 0.3 +
                min(macd_momentum, 1.0) * 0.3
            )
            
            return max(0.0, min(1.0, momentum_score))
            
        except:
            return 0.5
    
    def calculate_adaptive_levels(self, analysis: Dict) -> Dict:
        """Calculate adaptive stop loss and targets based on volatility"""
        try:
            current_price = analysis['current_price']
            volatility_pct = analysis['volatility_pct']
            change_24h = analysis['change_24h']
            
            # Adaptive stop loss (based on ATR and volatility)
            base_stop = max(
                self.config['stop_loss_min'],
                min(volatility_pct * 0.6, self.config['stop_loss_max'])
            )
            
            # Adjust stop based on recent performance
            if change_24h > 10:  # Very strong gainer, tighter stop
                stop_pct = base_stop * 0.8
            elif change_24h > 5:  # Strong gainer
                stop_pct = base_stop * 0.9
            else:  # Moderate gainer
                stop_pct = base_stop
            
            # Calculate targets
            quick_target_pct = max(stop_pct * 1.8, self.config['target_quick_min'])
            stretch_target_pct = max(stop_pct * 3.0, self.config['target_stretch_min'])
            
            # Price levels
            stop_loss = current_price * (1 - stop_pct / 100)
            target_quick = current_price * (1 + quick_target_pct / 100)
            target_stretch = current_price * (1 + stretch_target_pct / 100)
            
            # Risk/Reward ratio
            risk_reward_ratio = quick_target_pct / stop_pct
            
            return {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target_quick': target_quick,
                'target_stretch': target_stretch,
                'stop_pct': stop_pct,
                'quick_target_pct': quick_target_pct,
                'stretch_target_pct': stretch_target_pct,
                'risk_reward_ratio': risk_reward_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate levels: {e}")
            return {}
    
    def calculate_ml_validation(self, analysis: Dict) -> Tuple[float, float]:
        """Get ML validation score and confidence if models available"""
        if not self.models or not self.feature_columns:
            # Return based on momentum and technical indicators
            momentum_score = analysis.get('momentum_score', 0.5)
            volume_ratio = analysis.get('volume_ratio', 1.0)
            change_24h = analysis.get('change_24h', 0)
            
            # Simple scoring based on available metrics
            score = (
                momentum_score * 0.4 +
                min(volume_ratio / 3.0, 1.0) * 0.3 +
                min(abs(change_24h) / 10.0, 1.0) * 0.3
            )
            
            confidence = score * 0.8  # Conservative confidence without ML
            
            return max(0.0, min(1.0, score)), max(0.0, min(1.0, confidence))
        
        try:
            # Create feature vector (simplified)
            features = {}
            
            # Basic features that should exist
            basic_features = [
                'rsi', 'change_24h', 'change_1h', 'volume_ratio', 
                'momentum_score', 'volatility_pct'
            ]
            
            for feature in basic_features:
                if feature in analysis:
                    features[feature] = analysis[feature]
                else:
                    features[feature] = 0.0
            
            # Create DataFrame with available features
            feature_df = pd.DataFrame([features])
            
            # Align with model columns
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0
            
            feature_df = feature_df[self.feature_columns]
            
            # Get predictions from all models
            predictions = []
            for model in self.models:
                try:
                    pred = model.predict_proba(feature_df)[0]
                    if len(pred) > 1:
                        predictions.append(pred[1])  # Positive class probability
                    else:
                        predictions.append(pred[0])
                except:
                    predictions.append(0.5)
            
            # Calculate weighted average
            if predictions:
                ml_score = np.average(predictions, weights=self.model_weights[:len(predictions)])
                ml_confidence = 1.0 - np.std(predictions)  # Lower std = higher confidence
            else:
                ml_score = 0.5
                ml_confidence = 0.3
            
            return max(0.0, min(1.0, ml_score)), max(0.0, min(1.0, ml_confidence))
            
        except Exception as e:
            self.logger.error(f"ML validation failed: {e}")
            return 0.5, 0.3
    
    def determine_signal_grade(self, analysis: Dict, levels: Dict, 
                              ml_score: float, ml_confidence: float) -> SignalGrade:
        """Determine signal grade based on all factors"""
        try:
            change_24h = analysis['change_24h']
            momentum_score = analysis['momentum_score']
            volume_ratio = analysis['volume_ratio']
            risk_reward = levels['risk_reward_ratio']
            
            # Top gainer criteria
            if (change_24h > 8 and momentum_score > 0.7 and 
                volume_ratio > 2.0 and risk_reward > 2.0 and ml_score > 0.6):
                return SignalGrade.TOP_GAINER
            
            # Strong mover criteria
            elif (change_24h > 4 and momentum_score > 0.5 and 
                  volume_ratio > 1.5 and risk_reward > 1.8):
                return SignalGrade.STRONG_MOVER
            
            # Momentum criteria
            elif (change_24h > 2 and momentum_score > 0.4 and 
                  volume_ratio > 1.2 and risk_reward > 1.5):
                return SignalGrade.MOMENTUM
            
            # Speculative (high risk/reward but lower confidence)
            else:
                return SignalGrade.SPECULATIVE
                
        except:
            return SignalGrade.SPECULATIVE
    
    def calculate_position_size(self, grade: SignalGrade, analysis: Dict, 
                               levels: Dict) -> float:
        """Calculate position size based on grade and risk"""
        try:
            base_size = self.config['base_position_size']
            
            # Grade multipliers
            multipliers = {
                SignalGrade.TOP_GAINER: 2.5,
                SignalGrade.STRONG_MOVER: 2.0,
                SignalGrade.MOMENTUM: 1.5,
                SignalGrade.SPECULATIVE: 1.0
            }
            
            # Risk adjustment
            stop_pct = levels.get('stop_pct', 2.0)
            risk_adjustment = 2.0 / max(stop_pct, 0.5)  # Smaller position for larger stops
            
            # Volume adjustment
            volume_ratio = analysis.get('volume_ratio', 1.0)
            volume_adjustment = min(1.5, volume_ratio / 2.0)
            
            position_size = base_size * multipliers.get(grade, 1.0) * risk_adjustment * volume_adjustment
            
            return min(position_size, self.config['max_position_size'])
            
        except:
            return self.config['base_position_size']
    
    def scan_top_gainers(self) -> Dict:
        """Main scanning function - find top gainers with ML validation"""
        self.logger.info("🚀 Starting Elite Gainer Scan...")
        start_time = time.time()
        
        # Get symbols to scan
        symbols = self.get_top_symbols(self.config['max_symbols_to_scan'])
        
        if not symbols:
            self.logger.error("No symbols found to scan")
            return {}
        
        self.logger.info(f"Scanning {len(symbols)} symbols for top gainers...")
        
        # Analyze all symbols
        all_signals = []
        processed = 0
        
        for symbol in symbols:
            try:
                processed += 1
                if processed % 50 == 0:
                    self.logger.info(f"Processed {processed}/{len(symbols)} symbols...")
                
                # Get gainer metrics
                analysis = self.calculate_gainer_metrics(symbol)
                if not analysis:
                    continue
                
                # Filter for minimum requirements
                if (analysis['change_24h'] < self.config['min_24h_change'] or
                    analysis['volume_24h'] < self.config['min_volume_24h']):
                    continue
                
                # Calculate trading levels
                levels = self.calculate_adaptive_levels(analysis)
                if not levels:
                    continue
                
                # Get ML validation
                ml_score, ml_confidence = self.calculate_ml_validation(analysis)
                
                # Filter by ML requirements
                if (ml_score < self.config['min_ml_score'] or 
                    ml_confidence < self.config['min_ml_confidence']):
                    continue
                
                # Determine grade
                grade = self.determine_signal_grade(analysis, levels, ml_score, ml_confidence)
                
                # Calculate position size
                position_size = self.calculate_position_size(grade, analysis, levels)
                
                # Create signal
                signal = GainerSignal(
                    symbol=symbol,
                    grade=grade,
                    change_24h=analysis['change_24h'],
                    change_1h=analysis['change_1h'],
                    change_5m=analysis['change_5m'],
                    current_price=analysis['current_price'],
                    volume_24h=analysis['volume_24h'],
                    volume_ratio=analysis['volume_ratio'],
                    ml_score=ml_score,
                    ml_confidence=ml_confidence,
                    entry_price=levels['entry_price'],
                    stop_loss=levels['stop_loss'],
                    target_quick=levels['target_quick'],
                    target_stretch=levels['target_stretch'],
                    position_size_pct=position_size,
                    risk_pct=levels['stop_pct'],
                    reward_potential_pct=levels['stretch_target_pct'],
                    risk_reward_ratio=levels['risk_reward_ratio'],
                    momentum_score=analysis['momentum_score'],
                    volatility_score=analysis['volatility_pct'] / 10.0,  # Normalized
                    liquidity_score=analysis['liquidity_score'],
                    timestamp=datetime.now()
                )
                
                all_signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Sort signals by combined score
        def signal_score(signal):
            return (
                signal.change_24h * 0.3 +
                signal.ml_score * 100 * 0.3 +
                signal.momentum_score * 100 * 0.2 +
                signal.risk_reward_ratio * 10 * 0.2
            )
        
        all_signals.sort(key=signal_score, reverse=True)
        
        # Apply portfolio risk management
        portfolio_risk = 0.0
        final_signals = []
        
        for signal in all_signals:
            # Calculate risk contribution
            risk_contribution = (signal.position_size_pct / 100) * (signal.risk_pct / 100)
            
            if portfolio_risk + risk_contribution <= self.config['max_total_risk'] / 100:
                final_signals.append(signal)
                portfolio_risk += risk_contribution
                
                if len(final_signals) >= self.config['max_signals_output']:
                    break
        
        # Create report
        scan_time = time.time() - start_time
        
        report = {
            'scan_info': {
                'total_symbols_scanned': len(symbols),
                'signals_generated': len(final_signals),
                'scan_time_seconds': scan_time,
                'timestamp': datetime.now().isoformat(),
                'scanner_type': 'Elite Gainer Scanner',
                'focus': 'Top Gainers with ML Validation'
            },
            'portfolio_metrics': {
                'total_signals': len(final_signals),
                'total_portfolio_risk_pct': portfolio_risk * 100,
                'avg_risk_reward_ratio': np.mean([s.risk_reward_ratio for s in final_signals]) if final_signals else 0,
                'total_position_allocation_pct': sum([s.position_size_pct for s in final_signals]),
                'expected_return_pct': sum([s.position_size_pct * s.reward_potential_pct / 100 for s in final_signals]),
            },
            'top_signals': []
        }
        
        # Convert signals to dict format
        for i, signal in enumerate(final_signals, 1):
            signal_dict = {
                'rank': i,
                'symbol': signal.symbol,
                'grade': signal.grade.value,
                'change_24h': signal.change_24h,
                'change_1h': signal.change_1h,
                'current_price': signal.current_price,
                'volume_24h': signal.volume_24h,
                'ml_score': signal.ml_score,
                'ml_confidence': signal.ml_confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'target_quick': signal.target_quick,
                'target_stretch': signal.target_stretch,
                'position_size_pct': signal.position_size_pct,
                'risk_pct': signal.risk_pct,
                'reward_potential_pct': signal.reward_potential_pct,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'momentum_score': signal.momentum_score,
                'strategy_type': 'gainer_momentum',
                'timeframe': 'intraday',
                'catalyst': f"24h: +{signal.change_24h:.1f}% | ML: {signal.ml_score:.3f}"
            }
            report['top_signals'].append(signal_dict)
        
        self.logger.info(f"🚀 Scan complete: {len(final_signals)} elite gainer signals generated in {scan_time:.1f}s")
        
        return report

def main():
    """Run the Elite Gainer Scanner"""
    scanner = EliteGainerScanner()
    
    try:
        # Run the scan
        report = scanner.scan_top_gainers()
        
        # Save results
        os.makedirs("reports", exist_ok=True)
        
        with open("reports/elite_gainer_signals.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("🚀 ELITE GAINER SCANNER RESULTS 🚀")
        print("="*80)
        
        scan_info = report['scan_info']
        portfolio = report['portfolio_metrics']
        
        print(f"📊 Scanned: {scan_info['total_symbols_scanned']} symbols")
        print(f"🎯 Generated: {scan_info['signals_generated']} elite gainer signals")
        print(f"⏱️ Scan time: {scan_info['scan_time_seconds']:.1f} seconds")
        print(f"💰 Expected return: {portfolio['expected_return_pct']:.2f}%")
        print(f"🛡️ Portfolio risk: {portfolio['total_portfolio_risk_pct']:.2f}%")
        print(f"⚖️ Avg R/R ratio: {portfolio['avg_risk_reward_ratio']:.2f}")
        
        if report['top_signals']:
            print("\n🏆 TOP 5 GAINER SIGNALS:")
            for signal in report['top_signals'][:5]:
                print(f"{signal['rank']}. {signal['symbol']} - {signal['grade'].upper()}")
                print(f"   📈 24h: +{signal['change_24h']:.1f}% | 1h: +{signal['change_1h']:.1f}%")
                print(f"   💎 ML Score: {signal['ml_score']:.3f} | R/R: {signal['risk_reward_ratio']:.2f}")
                print(f"   🎯 Entry: ${signal['entry_price']:.6f} | Target: ${signal['target_quick']:.6f}")
                print()
        
        print("🚀 READY FOR DAY TRADING! 🚀")
        
    except Exception as e:
        print(f"❌ Scanner failed: {e}")
        raise

if __name__ == "__main__":
    main()
