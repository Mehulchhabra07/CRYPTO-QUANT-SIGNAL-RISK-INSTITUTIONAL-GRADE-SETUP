"""
⚡ PROFESSIONAL VOLATILITY REGIME SCANNER ⚡
Advanced regime detection and volatility-based trading strategies
Built for MAXIMUM PROFIT in different market conditions

Features:
- Multi-timeframe regime detection
- Volatility clustering analysis
- Mean reversion vs momentum classification
- Professional risk-adjusted position sizing
- Dynamic strategy selection based on market regime
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import ccxt
import time
import logging
from typing import Dict, List, Tuple, Optional
import os
import sys
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MOMENTUM_BREAKOUT = "momentum_breakout"

class VolatilityRegimeScanner:
    """
    Professional Volatility & Regime Detection Scanner
    Optimized for profit in different market conditions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize exchange
        try:
            self.exchange = ccxt.binance({
                'apiKey': '',
                'apiSecret': '',
                'sandbox': False,
                'rateLimit': 1200,
                'options': {'defaultType': 'spot'}
            })
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {e}")
            self.exchange = None
            
        # PROFESSIONAL PARAMETERS
        self.config = {
            # Regime Detection
            'trend_lookback': 50,              # 50-period trend detection
            'volatility_lookback': 30,         # 30-period volatility window
            'momentum_threshold': 0.05,        # 5% momentum threshold
            'volatility_threshold_high': 0.08, # 8% high volatility
            'volatility_threshold_low': 0.02,  # 2% low volatility
            
            # Strategy Parameters
            'mean_reversion_rsi_oversold': 25,
            'mean_reversion_rsi_overbought': 75,
            'momentum_rsi_bullish': 55,
            'trending_ema_periods': [12, 26, 50],
            'breakout_donchian_period': 20,
            
            # Risk Management
            'max_position_size': 0.15,         # 15% max position
            'volatility_position_scaling': True,
            'regime_position_multipliers': {
                MarketRegime.TRENDING_BULL: 1.5,
                MarketRegime.MOMENTUM_BREAKOUT: 2.0,
                MarketRegime.TRENDING_BEAR: 0.3,
                MarketRegime.HIGH_VOLATILITY: 0.5,
                MarketRegime.LOW_VOLATILITY: 0.8,
                MarketRegime.MEAN_REVERTING: 1.0,
            },
            
            # Quality Filters
            'min_volume_usd': 2000000,         # $2M minimum volume
            'min_price': 0.001,
            'max_spread': 0.003,               # 0.3% max spread
            'min_market_cap_rank': 150,
        }
        
        self.logger.info("⚡ Professional Volatility Regime Scanner initialized")
    
    def fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data across multiple timeframes for regime analysis"""
        timeframes = {
            '1h': 168,   # 1 week of hourly data
            '4h': 180,   # 30 days of 4h data  
            '1d': 100    # 100 days of daily data
        }
        
        data = {}
        
        for tf, limit in timeframes.items():
            try:
                if not self.exchange:
                    continue
                    
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                data[tf] = df
                
                time.sleep(0.05)  # Rate limiting
                
            except Exception as e:
                self.logger.debug(f"Failed to fetch {symbol} {tf}: {e}")
                continue
                
        return data
    
    def detect_regime(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Advanced multi-timeframe regime detection"""
        try:
            # Use 4h data as primary
            primary_df = data.get('4h')
            daily_df = data.get('1d')
            
            if primary_df is None or len(primary_df) < 50:
                return {'regime': MarketRegime.MEAN_REVERTING, 'confidence': 0.0}
            
            # Calculate regime indicators
            close = primary_df['close']
            
            # Trend indicators
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            ema_50 = close.ewm(span=50).mean()
            
            # Current trend strength
            trend_alignment = 0
            current_price = close.iloc[-1]
            
            if current_price > ema_12.iloc[-1] > ema_26.iloc[-1] > ema_50.iloc[-1]:
                trend_alignment = 2  # Strong bullish
            elif current_price > ema_12.iloc[-1] > ema_26.iloc[-1]:
                trend_alignment = 1  # Moderate bullish
            elif current_price < ema_12.iloc[-1] < ema_26.iloc[-1] < ema_50.iloc[-1]:
                trend_alignment = -2  # Strong bearish
            elif current_price < ema_12.iloc[-1] < ema_26.iloc[-1]:
                trend_alignment = -1  # Moderate bearish
            else:
                trend_alignment = 0  # Sideways
            
            # Volatility analysis
            returns = close.pct_change().dropna()
            volatility_30 = returns.rolling(30).std().iloc[-1] * np.sqrt(365)
            volatility_10 = returns.rolling(10).std().iloc[-1] * np.sqrt(365)
            
            # Momentum analysis
            momentum_20 = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
            momentum_5 = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
            
            # Volume analysis
            volume_ma = primary_df['volume'].rolling(20).mean()
            volume_ratio = primary_df['volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
            
            # RSI for mean reversion
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Regime classification logic
            regime, confidence = self._classify_regime(
                trend_alignment, volatility_30, volatility_10, 
                momentum_20, momentum_5, volume_ratio, current_rsi
            )
            
            return {
                'regime': regime,
                'confidence': confidence,
                'trend_alignment': trend_alignment,
                'volatility_30': volatility_30,
                'volatility_10': volatility_10,
                'momentum_20': momentum_20,
                'momentum_5': momentum_5,
                'volume_ratio': volume_ratio,
                'rsi': current_rsi,
                'regime_score': self._calculate_regime_score(regime, confidence, trend_alignment, momentum_20)
            }
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return {'regime': MarketRegime.MEAN_REVERTING, 'confidence': 0.0, 'regime_score': 0.0}
    
    def _classify_regime(self, trend_align: int, vol_30: float, vol_10: float, 
                        mom_20: float, mom_5: float, vol_ratio: float, rsi: float) -> Tuple[MarketRegime, float]:
        """Classify market regime with confidence"""
        
        # High volatility regime
        if vol_30 > self.config['volatility_threshold_high'] or vol_10 > self.config['volatility_threshold_high']:
            if vol_ratio > 2.0 and abs(mom_5) > 0.05:
                return MarketRegime.MOMENTUM_BREAKOUT, 0.9
            else:
                return MarketRegime.HIGH_VOLATILITY, 0.8
        
        # Low volatility regime
        if vol_30 < self.config['volatility_threshold_low'] and vol_10 < self.config['volatility_threshold_low']:
            return MarketRegime.LOW_VOLATILITY, 0.8
        
        # Trending regimes
        if abs(trend_align) >= 2:  # Strong trend alignment
            if mom_20 > self.config['momentum_threshold'] and trend_align > 0:
                return MarketRegime.TRENDING_BULL, 0.9
            elif mom_20 < -self.config['momentum_threshold'] and trend_align < 0:
                return MarketRegime.TRENDING_BEAR, 0.9
        
        # Moderate trending
        if abs(trend_align) == 1:
            if mom_20 > 0 and trend_align > 0:
                return MarketRegime.TRENDING_BULL, 0.6
            elif mom_20 < 0 and trend_align < 0:
                return MarketRegime.TRENDING_BEAR, 0.6
        
        # Mean reverting conditions
        if (rsi > 75 or rsi < 25) and abs(mom_20) < 0.02:
            return MarketRegime.MEAN_REVERTING, 0.7
        
        # Default to mean reverting with low confidence
        return MarketRegime.MEAN_REVERTING, 0.4
    
    def _calculate_regime_score(self, regime: MarketRegime, confidence: float, 
                               trend_align: int, momentum: float) -> float:
        """Calculate regime-based opportunity score"""
        
        base_score = confidence
        
        # Boost score for favorable regimes
        if regime in [MarketRegime.TRENDING_BULL, MarketRegime.MOMENTUM_BREAKOUT]:
            base_score *= 1.3
        elif regime == MarketRegime.TRENDING_BEAR:
            base_score *= 0.5  # Reduce for bear trends in crypto
        elif regime == MarketRegime.HIGH_VOLATILITY:
            base_score *= 0.8  # Reduce for high volatility
        
        # Trend strength bonus
        if abs(trend_align) >= 2:
            base_score *= 1.2
        
        # Momentum bonus
        if abs(momentum) > 0.08:  # Strong momentum
            base_score *= 1.1
        
        return min(1.0, base_score)
    
    def calculate_regime_strategy(self, regime_data: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate strategy and signals based on detected regime"""
        try:
            regime = regime_data.get('regime', MarketRegime.MEAN_REVERTING)
            confidence = regime_data.get('confidence', 0.0)
            
            close = price_data['close']
            high = price_data['high']
            low = price_data['low']
            volume = price_data['volume']
            
            current_price = close.iloc[-1]
            
            # Calculate ATR for stop losses
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Strategy-specific calculations
            if regime == MarketRegime.TRENDING_BULL:
                return self._trending_bull_strategy(close, atr, current_price, confidence)
            elif regime == MarketRegime.MOMENTUM_BREAKOUT:
                return self._momentum_breakout_strategy(close, high, low, atr, current_price, confidence)
            elif regime == MarketRegime.MEAN_REVERTING:
                return self._mean_reversion_strategy(close, atr, current_price, confidence, regime_data.get('rsi', 50))
            elif regime == MarketRegime.LOW_VOLATILITY:
                return self._low_volatility_strategy(close, atr, current_price, confidence)
            else:  # HIGH_VOLATILITY, TRENDING_BEAR
                return self._defensive_strategy(close, atr, current_price, confidence)
                
        except Exception as e:
            self.logger.error(f"Strategy calculation failed: {e}")
            return {}
    
    def _trending_bull_strategy(self, close: pd.Series, atr: float, price: float, confidence: float) -> Dict:
        """Trending bull market strategy"""
        # Pullback entry strategy
        ema_20 = close.ewm(span=20).mean().iloc[-1]
        
        # Entry: Current price (or slight pullback)
        entry_price = price
        
        # Stop: Below EMA20 or 2x ATR
        stop_loss = min(entry_price - (atr * 2.0), ema_20 * 0.98)
        
        # Targets: Conservative and aggressive
        target_1 = entry_price + (atr * 3.0)
        target_2 = entry_price + (atr * 5.0)
        
        # Position size: Larger in trending markets
        base_size = 0.12 * confidence
        regime_multiplier = self.config['regime_position_multipliers'].get(MarketRegime.TRENDING_BULL, 1.0)
        position_size = min(self.config['max_position_size'], base_size * regime_multiplier)
        
        return {
            'strategy': 'trending_bull_pullback',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'position_size_pct': position_size * 100,
            'risk_reward_1': (target_1 - entry_price) / (entry_price - stop_loss),
            'risk_reward_2': (target_2 - entry_price) / (entry_price - stop_loss),
            'confidence': confidence,
            'hold_time_days': 5
        }
    
    def _momentum_breakout_strategy(self, close: pd.Series, high: pd.Series, low: pd.Series, 
                                   atr: float, price: float, confidence: float) -> Dict:
        """Momentum breakout strategy"""
        # Donchian breakout levels
        donchian_high = high.rolling(self.config['breakout_donchian_period']).max().iloc[-1]
        donchian_low = low.rolling(self.config['breakout_donchian_period']).min().iloc[-1]
        
        # Entry: Current price (assuming near breakout)
        entry_price = price
        
        # Stop: Recent swing low or 1.5x ATR
        stop_loss = max(donchian_low, entry_price - (atr * 1.5))
        
        # Targets: Aggressive for breakouts
        target_1 = entry_price + (atr * 4.0)
        target_2 = entry_price + (atr * 7.0)
        
        # Position size: Largest for breakouts
        base_size = 0.10 * confidence
        regime_multiplier = self.config['regime_position_multipliers'].get(MarketRegime.MOMENTUM_BREAKOUT, 1.0)
        position_size = min(self.config['max_position_size'], base_size * regime_multiplier)
        
        return {
            'strategy': 'momentum_breakout',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'position_size_pct': position_size * 100,
            'risk_reward_1': (target_1 - entry_price) / (entry_price - stop_loss),
            'risk_reward_2': (target_2 - entry_price) / (entry_price - stop_loss),
            'confidence': confidence,
            'hold_time_days': 3
        }
    
    def _mean_reversion_strategy(self, close: pd.Series, atr: float, price: float, 
                                confidence: float, rsi: float) -> Dict:
        """Mean reversion strategy"""
        # Mean reversion levels
        sma_20 = close.rolling(20).mean().iloc[-1]
        
        # Entry: Current price
        entry_price = price
        
        # Stop: Tighter for mean reversion
        stop_loss = entry_price - (atr * 1.5)
        
        # Targets: Conservative for mean reversion
        target_1 = sma_20 if sma_20 > entry_price else entry_price + (atr * 2.0)
        target_2 = entry_price + (atr * 3.0)
        
        # Position size: Moderate
        base_size = 0.08 * confidence
        regime_multiplier = self.config['regime_position_multipliers'].get(MarketRegime.MEAN_REVERTING, 1.0)
        position_size = min(self.config['max_position_size'], base_size * regime_multiplier)
        
        return {
            'strategy': 'mean_reversion',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'position_size_pct': position_size * 100,
            'risk_reward_1': (target_1 - entry_price) / (entry_price - stop_loss),
            'risk_reward_2': (target_2 - entry_price) / (entry_price - stop_loss),
            'confidence': confidence,
            'hold_time_days': 4
        }
    
    def _low_volatility_strategy(self, close: pd.Series, atr: float, price: float, confidence: float) -> Dict:
        """Low volatility accumulation strategy"""
        entry_price = price
        stop_loss = entry_price - (atr * 2.5)  # Wider stops in low vol
        target_1 = entry_price + (atr * 2.5)   # Conservative targets
        target_2 = entry_price + (atr * 4.0)
        
        base_size = 0.10 * confidence
        regime_multiplier = self.config['regime_position_multipliers'].get(MarketRegime.LOW_VOLATILITY, 1.0)
        position_size = min(self.config['max_position_size'], base_size * regime_multiplier)
        
        return {
            'strategy': 'low_volatility_accumulation',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'position_size_pct': position_size * 100,
            'risk_reward_1': (target_1 - entry_price) / (entry_price - stop_loss),
            'risk_reward_2': (target_2 - entry_price) / (entry_price - stop_loss),
            'confidence': confidence,
            'hold_time_days': 7
        }
    
    def _defensive_strategy(self, close: pd.Series, atr: float, price: float, confidence: float) -> Dict:
        """Defensive strategy for unfavorable conditions"""
        entry_price = price
        stop_loss = entry_price - (atr * 1.0)  # Tight stops
        target_1 = entry_price + (atr * 1.5)   # Quick profits
        target_2 = entry_price + (atr * 2.5)
        
        # Very small position size
        position_size = 0.03 * confidence
        
        return {
            'strategy': 'defensive_scalp',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'position_size_pct': position_size * 100,
            'risk_reward_1': (target_1 - entry_price) / (entry_price - stop_loss),
            'risk_reward_2': (target_2 - entry_price) / (entry_price - stop_loss),
            'confidence': confidence,
            'hold_time_days': 1
        }
    
    def scan_symbol_regime(self, symbol: str) -> Optional[Dict]:
        """Scan symbol with regime-based analysis"""
        try:
            # Fetch multi-timeframe data
            data = self.fetch_multi_timeframe_data(symbol)
            
            if not data or '4h' not in data:
                return None
            
            primary_df = data['4h']
            
            # Basic filters
            current_price = primary_df['close'].iloc[-1]
            if current_price < self.config['min_price']:
                return None
            
            volume_usd = primary_df['volume'].iloc[-1] * current_price
            if volume_usd < self.config['min_volume_usd']:
                return None
            
            # Detect regime
            regime_data = self.detect_regime(data)
            
            # Calculate strategy
            strategy_data = self.calculate_regime_strategy(regime_data, primary_df)
            
            if not strategy_data:
                return None
            
            # Quality score
            regime_score = regime_data.get('regime_score', 0.0)
            confidence = regime_data.get('confidence', 0.0)
            
            # Only high-quality opportunities
            if regime_score < 0.6 or confidence < 0.5:
                return None
            
            # Combined result
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'current_price': current_price,
                'volume_usd': volume_usd,
                
                # Regime analysis
                'regime': regime_data.get('regime').value,
                'regime_confidence': confidence,
                'regime_score': regime_score,
                'trend_alignment': regime_data.get('trend_alignment', 0),
                'volatility_30': regime_data.get('volatility_30', 0),
                'momentum_20': regime_data.get('momentum_20', 0),
                'rsi': regime_data.get('rsi', 50),
                
                # Strategy execution
                **strategy_data,
                
                # Risk metrics
                'risk_pct': ((strategy_data['entry_price'] - strategy_data['stop_loss']) / strategy_data['entry_price']) * 100,
                'max_portfolio_risk': (strategy_data['position_size_pct'] / 100) * ((strategy_data['entry_price'] - strategy_data['stop_loss']) / strategy_data['entry_price']) * 100,
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Regime scan failed for {symbol}: {e}")
            return None
    
    def scan_universe_regime(self, limit: int = 50) -> Dict:
        """Scan crypto universe with regime-based analysis"""
        self.logger.info(f"⚡ Scanning {limit} cryptos for REGIME-BASED opportunities...")
        
        # Get top symbols
        try:
            if not self.exchange:
                self.logger.error("❌ Exchange not available")
                return {'error': 'Exchange not available'}
            
            markets = self.exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            
            # Get volume data
            volume_data = []
            for symbol in usdt_pairs[:150]:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    volume_usd = ticker.get('quoteVolume', 0)
                    volume_data.append((symbol, volume_usd))
                except:
                    continue
            
            volume_data.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, _ in volume_data[:limit]]
            
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            top_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        
        # Scan symbols
        opportunities = []
        regime_stats = {}
        
        for i, symbol in enumerate(top_symbols):
            try:
                result = self.scan_symbol_regime(symbol)
                if result:
                    opportunities.append(result)
                    regime = result['regime']
                    regime_stats[regime] = regime_stats.get(regime, 0) + 1
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"📊 Scanned {i+1}/{len(top_symbols)}, found {len(opportunities)} opportunities")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by regime score
        opportunities.sort(key=lambda x: x['regime_score'], reverse=True)
        
        # Create report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'scanner': 'Professional Volatility & Regime Scanner',
            'symbols_scanned': len(top_symbols),
            'opportunities_found': len(opportunities),
            'success_rate_pct': (len(opportunities) / len(top_symbols)) * 100,
            'regime_distribution': regime_stats,
            'top_opportunities': opportunities[:15],
            'market_overview': self._generate_market_overview(opportunities),
            'strategy_allocation': self._generate_strategy_allocation(opportunities)
        }
        
        self.logger.info(f"✅ Regime scan complete: {len(opportunities)} opportunities found!")
        
        return report
    
    def _generate_market_overview(self, opportunities: List[Dict]) -> Dict:
        """Generate market overview from opportunities"""
        if not opportunities:
            return {}
        
        total_regime_score = sum(opp['regime_score'] for opp in opportunities)
        avg_regime_score = total_regime_score / len(opportunities)
        
        volatility_levels = [opp['volatility_30'] for opp in opportunities]
        avg_volatility = sum(volatility_levels) / len(volatility_levels)
        
        momentum_levels = [opp['momentum_20'] for opp in opportunities]
        avg_momentum = sum(momentum_levels) / len(momentum_levels)
        
        return {
            'average_regime_score': avg_regime_score,
            'average_volatility': avg_volatility,
            'average_momentum': avg_momentum,
            'market_sentiment': 'bullish' if avg_momentum > 0.02 else 'bearish' if avg_momentum < -0.02 else 'neutral'
        }
    
    def _generate_strategy_allocation(self, opportunities: List[Dict]) -> Dict:
        """Generate optimal strategy allocation"""
        if not opportunities:
            return {}
        
        strategy_counts = {}
        total_position_size = 0
        
        for opp in opportunities[:10]:  # Top 10 opportunities
            strategy = opp.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            total_position_size += opp.get('position_size_pct', 0)
        
        return {
            'strategy_distribution': strategy_counts,
            'total_portfolio_allocation': total_position_size,
            'recommended_max_positions': min(10, len(opportunities)),
            'primary_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'none'
        }

def run_regime_scan():
    """Run the professional regime-based scanner"""
    scanner = VolatilityRegimeScanner()
    report = scanner.scan_universe_regime(limit=40)
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/regime_volatility_scan.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"⚡ PROFESSIONAL REGIME SCAN COMPLETE!")
    if 'top_opportunities' in report:
        print(f"📊 Found {len(report['top_opportunities'])} regime-based opportunities")
        print(f"💰 Success rate: {report.get('success_rate_pct', 0):.1f}%")
        
        if report['top_opportunities']:
            print(f"\n🏆 TOP 5 REGIME OPPORTUNITIES:")
            for i, opp in enumerate(report['top_opportunities'][:5], 1):
                print(f"  {i}. {opp['symbol']}: {opp['regime_score']:.3f} score ({opp['regime']}, {opp['strategy']})")
    
    return report

if __name__ == "__main__":
    run_regime_scan()
