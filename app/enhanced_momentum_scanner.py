"""
🚀 ENHANCED MOMENTUM & BREAKOUT SCANNER 🚀
Advanced momentum detection and volatility breakout analysis
Designed to catch the BIG MOVES that make money

Features:
- Momentum regime detection
- Volatility compression/expansion cycles  
- Breakout pattern recognition
- High-probability setups only
- Professional execution levels
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)

class MomentumBreakoutScanner:
    """
    Enhanced scanner focused on MOMENTUM and BREAKOUT patterns
    Designed to catch explosive moves that generate real profits
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize exchange
        try:
            self.exchange = ccxt.binance({
                'apiKey': '',  # Read-only, no API key needed
                'apiSecret': '',
                'sandbox': False,
                'rateLimit': 1200,
                'options': {'defaultType': 'spot'}
            })
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {e}")
            self.exchange = None
            
        # MOMENTUM PARAMETERS - Optimized for explosive moves
        self.config = {
            # Momentum Detection
            'momentum_lookback': 20,           # 20-period momentum
            'momentum_threshold': 0.08,        # 8% minimum momentum
            'acceleration_periods': 5,         # 5-period acceleration  
            'volume_surge_threshold': 2.0,     # 2x volume surge
            
            # Volatility Breakout
            'volatility_lookback': 30,         # 30-period volatility
            'compression_threshold': 0.6,      # 60% volatility compression
            'expansion_threshold': 1.5,        # 150% volatility expansion
            'atr_breakout_multiplier': 2.5,    # 2.5x ATR breakout
            
            # Pattern Recognition
            'donchian_period': 20,             # 20-period Donchian
            'bollinger_period': 20,            # 20-period Bollinger
            'bollinger_std': 2.0,              # 2 standard deviations
            'squeeze_threshold': 0.8,          # Squeeze detection
            
            # Quality Filters
            'min_price': 0.001,                # Minimum price $0.001
            'min_volume': 1000000,             # Minimum $1M volume
            'max_spread': 0.005,               # Maximum 0.5% spread
            'min_market_cap_rank': 200,        # Top 200 by market cap
            
            # Risk Management
            'max_volatility': 0.15,            # 15% max daily volatility
            'stop_loss_atr': 2.0,              # 2x ATR stop loss
            'profit_target_atr': 4.0,          # 4x ATR profit target
            'position_size_volatility_adj': True,
        }
        
        self.logger.info("🚀 Momentum Breakout Scanner initialized")
    
    def fetch_market_data(self, symbol: str, timeframe: str = '4h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with error handling"""
        try:
            if not self.exchange:
                return None
                
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.debug(f"Failed to fetch {symbol}: {e}")
            return None
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate advanced momentum indicators"""
        try:
            # Price momentum
            momentum_20 = (df['close'].iloc[-1] / df['close'].iloc[-self.config['momentum_lookback']] - 1)
            momentum_10 = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1)
            momentum_5 = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
            
            # Momentum acceleration (is momentum increasing?)
            recent_momentum = momentum_5
            older_momentum = (df['close'].iloc[-5] / df['close'].iloc[-10] - 1)
            momentum_acceleration = recent_momentum - older_momentum
            
            # Volume momentum  
            volume_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
            
            # Price velocity (rate of change)
            velocity = df['close'].pct_change().rolling(5).mean().iloc[-1]
            
            # RSI momentum
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            return {
                'momentum_20': momentum_20,
                'momentum_10': momentum_10,
                'momentum_5': momentum_5,
                'momentum_acceleration': momentum_acceleration,
                'volume_ratio': volume_ratio,
                'velocity': velocity,
                'rsi': current_rsi,
                'momentum_score': self._calculate_momentum_score(momentum_20, momentum_acceleration, volume_ratio, current_rsi)
            }
            
        except Exception as e:
            self.logger.error(f"Momentum calculation failed: {e}")
            return {}
    
    def _calculate_momentum_score(self, momentum_20: float, acceleration: float, volume_ratio: float, rsi: float) -> float:
        """Calculate composite momentum score (0-1)"""
        
        # Momentum component (40% weight)
        momentum_component = min(1.0, max(0.0, (momentum_20 + 0.05) / 0.15))  # Normalize -5% to +10%
        
        # Acceleration component (30% weight)  
        accel_component = min(1.0, max(0.0, (acceleration + 0.02) / 0.04))  # Normalize -2% to +2%
        
        # Volume component (20% weight)
        volume_component = min(1.0, (volume_ratio - 1.0) / 2.0)  # Normalize 1x to 3x volume
        
        # RSI component (10% weight) - favor 50-70 range
        if 50 <= rsi <= 70:
            rsi_component = 1.0
        elif rsi > 70:
            rsi_component = max(0.0, 1.0 - (rsi - 70) / 30)
        else:
            rsi_component = max(0.0, rsi / 50)
            
        # Weighted score
        momentum_score = (
            momentum_component * 0.4 +
            accel_component * 0.3 +
            volume_component * 0.2 +
            rsi_component * 0.1
        )
        
        return min(1.0, max(0.0, momentum_score))
    
    def calculate_breakout_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility breakout indicators"""
        try:
            # ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Volatility compression/expansion
            volatility_30 = df['close'].pct_change().rolling(30).std().iloc[-1]
            volatility_10 = df['close'].pct_change().rolling(10).std().iloc[-1]
            volatility_ratio = volatility_10 / volatility_30 if volatility_30 > 0 else 1.0
            
            # Donchian breakout
            donchian_high = df['high'].rolling(self.config['donchian_period']).max().iloc[-1]
            donchian_low = df['low'].rolling(self.config['donchian_period']).min().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Distance to breakout levels
            breakout_high_distance = (donchian_high - current_price) / current_price
            breakout_low_distance = (current_price - donchian_low) / current_price
            
            # Bollinger squeeze detection
            bb_ma = df['close'].rolling(self.config['bollinger_period']).mean()
            bb_std = df['close'].rolling(self.config['bollinger_period']).std()
            bb_upper = bb_ma + (bb_std * self.config['bollinger_std'])
            bb_lower = bb_ma - (bb_std * self.config['bollinger_std'])
            bb_width = (bb_upper - bb_lower) / bb_ma
            
            # Keltner channels for squeeze
            kc_ma = df['close'].rolling(self.config['bollinger_period']).mean()
            kc_range = true_range.rolling(self.config['bollinger_period']).mean() * 1.5
            kc_upper = kc_ma + kc_range
            kc_lower = kc_ma - kc_range
            
            # Squeeze condition: Bollinger inside Keltner
            squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
            
            # Price position in range
            price_position = (current_price - donchian_low) / (donchian_high - donchian_low) if donchian_high > donchian_low else 0.5
            
            return {
                'atr': atr,
                'atr_pct': atr / current_price,
                'volatility_ratio': volatility_ratio,
                'donchian_high': donchian_high,
                'donchian_low': donchian_low,
                'breakout_high_distance': breakout_high_distance,
                'breakout_low_distance': breakout_low_distance,
                'squeeze': squeeze,
                'bb_width': bb_width.iloc[-1],
                'price_position': price_position,
                'breakout_score': self._calculate_breakout_score(volatility_ratio, squeeze, price_position, breakout_high_distance)
            }
            
        except Exception as e:
            self.logger.error(f"Breakout calculation failed: {e}")
            return {}
    
    def _calculate_breakout_score(self, vol_ratio: float, squeeze: bool, price_pos: float, breakout_dist: float) -> float:
        """Calculate composite breakout score (0-1)"""
        
        # Volatility expansion component (40% weight)
        vol_component = min(1.0, max(0.0, (vol_ratio - 0.8) / 0.7))  # Normalize 0.8 to 1.5
        
        # Squeeze component (30% weight)
        squeeze_component = 1.0 if squeeze else 0.3
        
        # Price position component (20% weight) - favor near highs for breakouts
        position_component = price_pos  # 0 = at low, 1 = at high
        
        # Breakout proximity component (10% weight) - closer to breakout = higher score
        proximity_component = max(0.0, 1.0 - abs(breakout_dist) * 20)  # Closer to breakout = higher score
        
        # Weighted score
        breakout_score = (
            vol_component * 0.4 +
            squeeze_component * 0.3 +
            position_component * 0.2 +
            proximity_component * 0.1
        )
        
        return min(1.0, max(0.0, breakout_score))
    
    def calculate_trading_levels(self, df: pd.DataFrame, momentum_data: Dict, breakout_data: Dict) -> Dict:
        """Calculate professional trading execution levels"""
        try:
            current_price = df['close'].iloc[-1]
            atr = breakout_data.get('atr', current_price * 0.02)
            
            # Entry price (current market)
            entry_price = current_price
            
            # Stop loss: ATR-based
            stop_loss = entry_price - (atr * self.config['stop_loss_atr'])
            
            # Profit targets: Multiple levels
            target_1 = entry_price + (atr * 2.0)  # Conservative
            target_2 = entry_price + (atr * self.config['profit_target_atr'])  # Aggressive
            
            # Position sizing: Based on volatility
            volatility = breakout_data.get('atr_pct', 0.02)
            base_position_size = 0.1  # 10% base
            
            if self.config['position_size_volatility_adj']:
                # Reduce position size in high volatility
                volatility_adj = max(0.5, 1.0 - (volatility - 0.02) * 10)
                position_size = base_position_size * volatility_adj
            else:
                position_size = base_position_size
                
            # Risk metrics
            risk_pct = (entry_price - stop_loss) / entry_price
            reward_1_pct = (target_1 - entry_price) / entry_price
            reward_2_pct = (target_2 - entry_price) / entry_price
            
            risk_reward_1 = reward_1_pct / risk_pct if risk_pct > 0 else 0
            risk_reward_2 = reward_2_pct / risk_pct if risk_pct > 0 else 0
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_1': target_1,
                'target_2': target_2,
                'position_size_pct': position_size * 100,
                'risk_pct': risk_pct * 100,
                'reward_1_pct': reward_1_pct * 100,
                'reward_2_pct': reward_2_pct * 100,
                'risk_reward_1': risk_reward_1,
                'risk_reward_2': risk_reward_2,
                'atr_value': atr
            }
            
        except Exception as e:
            self.logger.error(f"Trading levels calculation failed: {e}")
            return {}
    
    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """Scan a single symbol for momentum/breakout opportunities"""
        try:
            # Fetch data
            df = self.fetch_market_data(symbol, '4h', 100)
            if df is None or len(df) < 50:
                return None
                
            # Basic filters
            current_price = df['close'].iloc[-1]
            if current_price < self.config['min_price']:
                return None
                
            volume_usd = df['volume'].iloc[-1] * current_price
            if volume_usd < self.config['min_volume']:
                return None
                
            # Calculate indicators
            momentum_data = self.calculate_momentum_indicators(df)
            breakout_data = self.calculate_breakout_indicators(df)
            
            if not momentum_data or not breakout_data:
                return None
                
            # Calculate trading levels
            trading_levels = self.calculate_trading_levels(df, momentum_data, breakout_data)
            
            # Composite scoring
            momentum_score = momentum_data.get('momentum_score', 0)
            breakout_score = breakout_data.get('breakout_score', 0)
            
            # Combined score: 60% momentum, 40% breakout
            combined_score = (momentum_score * 0.6) + (breakout_score * 0.4)
            
            # Quality filters
            if combined_score < 0.6:  # Only high-quality setups
                return None
                
            if breakout_data.get('atr_pct', 0) > self.config['max_volatility']:
                return None
                
            # Create result
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'combined_score': combined_score,
                'momentum_score': momentum_score,
                'breakout_score': breakout_score,
                'current_price': current_price,
                'volume_usd': volume_usd,
                
                # Momentum metrics
                'momentum_20': momentum_data.get('momentum_20', 0),
                'momentum_acceleration': momentum_data.get('momentum_acceleration', 0),
                'volume_ratio': momentum_data.get('volume_ratio', 1),
                'rsi': momentum_data.get('rsi', 50),
                
                # Breakout metrics
                'volatility_ratio': breakout_data.get('volatility_ratio', 1),
                'squeeze': breakout_data.get('squeeze', False),
                'price_position': breakout_data.get('price_position', 0.5),
                'atr_pct': breakout_data.get('atr_pct', 0.02),
                
                # Trading execution
                **trading_levels,
                
                # Strategy classification
                'strategy_type': self._classify_strategy(momentum_score, breakout_score, breakout_data.get('squeeze', False)),
                'timeframe': '4h',
                'expected_hold_days': 3,
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Scan failed for {symbol}: {e}")
            return None
    
    def _classify_strategy(self, momentum_score: float, breakout_score: float, squeeze: bool) -> str:
        """Classify the primary strategy type"""
        if squeeze and breakout_score > 0.7:
            return "volatility_breakout"
        elif momentum_score > 0.8:
            return "momentum_continuation"
        elif momentum_score > 0.6 and breakout_score > 0.6:
            return "momentum_breakout_combo"
        else:
            return "hybrid_opportunity"
    
    def scan_crypto_universe(self, limit: int = 100) -> Dict:
        """Scan the crypto universe for momentum/breakout opportunities"""
        self.logger.info(f"🚀 Scanning {limit} cryptos for MOMENTUM & BREAKOUT opportunities...")
        
        # Get top cryptos by volume
        try:
            if not self.exchange:
                self.logger.error("❌ Exchange not available")
                return {'error': 'Exchange not available'}
                
            markets = self.exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            
            # Sort by volume and take top N
            volume_data = []
            for symbol in usdt_pairs[:200]:  # Check top 200 first
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    volume_usd = ticker.get('quoteVolume', 0)
                    volume_data.append((symbol, volume_usd))
                except:
                    continue
                    
            volume_data.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, _ in volume_data[:limit]]
            
        except Exception as e:
            self.logger.error(f"Failed to get top symbols: {e}")
            # Fallback list
            top_symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
                'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT'
            ]
        
        # Scan each symbol
        opportunities = []
        scanned = 0
        
        for symbol in top_symbols:
            try:
                result = self.scan_symbol(symbol)
                if result:
                    opportunities.append(result)
                
                scanned += 1
                if scanned % 10 == 0:
                    self.logger.info(f"📊 Scanned {scanned}/{len(top_symbols)} symbols, found {len(opportunities)} opportunities")
                    
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by combined score
        opportunities.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Create final report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'scanner': 'Enhanced Momentum & Breakout Scanner',
            'symbols_scanned': scanned,
            'opportunities_found': len(opportunities),
            'success_rate_pct': (len(opportunities) / max(scanned, 1)) * 100,
            'top_opportunities': opportunities[:20],  # Top 20
            'filters_applied': {
                'min_combined_score': 0.6,
                'min_volume_usd': self.config['min_volume'],
                'max_volatility': self.config['max_volatility'],
                'min_price': self.config['min_price']
            },
            'strategy_breakdown': self._analyze_strategy_breakdown(opportunities)
        }
        
        self.logger.info(f"✅ Scan complete: {len(opportunities)} momentum/breakout opportunities found!")
        
        return report
    
    def _analyze_strategy_breakdown(self, opportunities: List[Dict]) -> Dict:
        """Analyze the distribution of strategy types"""
        strategy_counts = {}
        total_score = 0
        
        for opp in opportunities:
            strategy = opp.get('strategy_type', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            total_score += opp.get('combined_score', 0)
            
        return {
            'strategy_distribution': strategy_counts,
            'average_score': total_score / len(opportunities) if opportunities else 0,
            'top_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'none'
        }

def run_enhanced_momentum_scan():
    """Run the enhanced momentum & breakout scanner"""
    scanner = MomentumBreakoutScanner()
    report = scanner.scan_crypto_universe(limit=50)
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/momentum_breakout_scan.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"🚀 MOMENTUM & BREAKOUT SCAN COMPLETE!")
    if 'top_opportunities' in report:
        print(f"📊 Found {len(report['top_opportunities'])} high-probability opportunities")
        print(f"💰 Success rate: {report.get('success_rate_pct', 0):.1f}%")
        
        if report['top_opportunities']:
            print(f"\n🏆 TOP 5 OPPORTUNITIES:")
            for i, opp in enumerate(report['top_opportunities'][:5], 1):
                print(f"  {i}. {opp['symbol']}: {opp['combined_score']:.3f} score ({opp['strategy_type']})")
    
    return report

if __name__ == "__main__":
    run_enhanced_momentum_scan()
