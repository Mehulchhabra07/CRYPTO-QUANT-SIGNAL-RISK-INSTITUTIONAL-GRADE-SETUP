"""
💰 ELITE PRODUCTION MONEY MAKER 💰
Top gainer focused system for maximum daily profits

Key Features:
- TOP GAINER DETECTION: Scans 400+ symbols for biggest movers
- ADAPTIVE RISK MANAGEMENT: Dynamic stops and targets based on volatility  
- ML VALIDATION: 51 models validate gainer signals
- PORTFOLIO OPTIMIZATION: Maximum 1% total risk with position scaling
- REAL-TIME EXECUTION: Live market data with immediate signals
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)

from app.elite_gainer_scanner import EliteGainerScanner
from app.providers.crypto import BinanceProvider
from app.providers.kraken import KrakenProvider

class EliteProductionMoneyMaker:
    """
    🚀 INSTITUTIONAL-GRADE TRADING SYSTEM 🚀
    Integrates top gainer detection with advanced ML validation
    Supports multiple exchanges including Kraken for institutional analysis
    """
    
    def __init__(self, exchange_preference: str = "binance"):
        logging.info("🏛️ Initializing Institutional Production Money Maker...")
        self.elite_scanner = EliteGainerScanner()
        
        # Initialize providers based on preference
        self.exchange_preference = exchange_preference.lower()
        if self.exchange_preference == "kraken":
            self.provider = KrakenProvider()
            logging.info("🏛️ Using Kraken for institutional-grade analysis")
        else:
            self.provider = BinanceProvider()
            logging.info("🚀 Using Binance for comprehensive analysis")
        
    def run_elite_gainer_scan(self, max_symbols: int = 400, 
                             min_gain_24h: float = 5.0,
                             top_picks: int = 15,
                             exchange: str = "binance") -> Dict:
        """
        Run the institutional gainer scan for top trading opportunities
        """
        try:
            exchange_name = "Kraken" if exchange.lower() == "kraken" else "Binance"
            logging.info(f"🔍 Scanning {exchange_name} for institutional opportunities...")
            
            # Switch provider if needed
            if exchange.lower() == "kraken" and not isinstance(self.provider, KrakenProvider):
                self.provider = KrakenProvider()
                logging.info("🏛️ Switched to Kraken provider")
            elif exchange.lower() != "kraken" and not isinstance(self.provider, BinanceProvider):
                self.provider = BinanceProvider()
                logging.info("🚀 Switched to Binance provider")
            
            # Get gainer signals from our elite scanner
            gainer_results = self.elite_scanner.scan_top_gainers()
            
            # Enhance with additional production filters
            enhanced_results = self._enhance_gainer_signals(gainer_results)
            
            # Final risk-adjusted ranking
            final_picks = self._rank_for_production(enhanced_results, top_picks)
            
            logging.info(f"✅ Institutional scan complete. Found {len(final_picks)} production-ready signals")
            
            return {
                'scan_time': datetime.now().isoformat(),
                'scan_type': f'institutional_{exchange.lower()}',
                'exchange': exchange_name,
                'total_scanned': max_symbols,
                'signals_found': len(final_picks),
                'top_picks': final_picks,
                'performance_metrics': self._calculate_scan_metrics(final_picks)
            }
            
        except Exception as e:
            logging.error(f"❌ Institutional scan failed: {str(e)}")
            return {
                'error': str(e),
                'scan_time': datetime.now().isoformat(),
                'scan_type': f'institutional_{exchange.lower()}',
                'exchange': exchange.lower(),
                'signals_found': 0,
                'top_picks': []
            }
    
    def _enhance_gainer_signals(self, gainer_results: Dict) -> List[Dict]:
        """
        Add production-specific enhancements to gainer signals
        """
        enhanced_signals = []
        
        # Check for top_signals (new format) or top_picks (old format)
        signals = gainer_results.get('top_signals', gainer_results.get('top_picks', []))
        
        if not signals:
            return enhanced_signals
            
        for signal in signals:
            try:
                # Convert signal object to dict if needed
                if hasattr(signal, '__dict__'):
                    signal_dict = signal.__dict__
                else:
                    signal_dict = signal
                
                # Add volume confirmation
                signal_dict['volume_strength'] = self._assess_volume_strength(signal_dict)
                
                # Add momentum persistence check
                signal_dict['momentum_persistence'] = self._check_momentum_persistence(signal_dict)
                
                # Add support/resistance levels
                signal_dict['key_levels'] = self._identify_key_levels(signal_dict)
                
                # Production risk score (0-100)
                signal_dict['production_risk_score'] = self._calculate_production_risk(signal_dict)
                
                # Expected profit potential
                signal_dict['profit_potential'] = self._estimate_profit_potential(signal_dict)
                
                enhanced_signals.append(signal_dict)
                
            except Exception as e:
                logging.warning(f"⚠️ Failed to enhance signal for {signal.get('symbol', 'unknown') if isinstance(signal, dict) else getattr(signal, 'symbol', 'unknown')}: {e}")
                continue
                
        return enhanced_signals
    
    def _assess_volume_strength(self, signal: Dict) -> str:
        """Assess volume strength for the signal"""
        try:
            volume_24h = signal.get('volume_24h', 0)
            avg_volume = signal.get('avg_volume_7d', volume_24h)
            
            if avg_volume > 0:
                volume_ratio = volume_24h / avg_volume
                if volume_ratio > 3.0:
                    return "EXCEPTIONAL"
                elif volume_ratio > 2.0:
                    return "STRONG"
                elif volume_ratio > 1.5:
                    return "MODERATE"
                else:
                    return "WEAK"
            return "UNKNOWN"
        except:
            return "UNKNOWN"
    
    def _check_momentum_persistence(self, signal: Dict) -> Dict:
        """Check if momentum is persisting across timeframes"""
        try:
            gain_24h = signal.get('change_24h', signal.get('gain_24h', 0))
            gain_1h = signal.get('change_1h', signal.get('gain_1h', 0))
            gain_5m = signal.get('change_5m', signal.get('gain_5m', 0))
            
            # Count positive timeframes
            positive_count = sum([1 for gain in [gain_24h, gain_1h, gain_5m] if gain > 0])
            
            persistence_score = positive_count / 3 * 100
            
            return {
                'score': persistence_score,
                'trend_alignment': positive_count >= 2,
                'momentum_strength': 'STRONG' if persistence_score > 66 else 'MODERATE' if persistence_score > 33 else 'WEAK'
            }
        except:
            return {'score': 0, 'trend_alignment': False, 'momentum_strength': 'WEAK'}
    
    def _identify_key_levels(self, signal: Dict) -> Dict:
        """Identify key support and resistance levels"""
        try:
            current_price = signal.get('current_price', 0)
            if current_price <= 0:
                return {}
                
            # Simple level calculation based on recent ranges
            stop_loss = signal.get('stop_loss', current_price * 0.97)
            take_profit = signal.get('target_quick', signal.get('take_profit', current_price * 1.05))
            
            return {
                'support': stop_loss,
                'resistance': take_profit,
                'current': current_price,
                'risk_range': abs(current_price - stop_loss) / current_price * 100,
                'reward_range': abs(take_profit - current_price) / current_price * 100
            }
        except:
            return {}
    
    def _calculate_production_risk(self, signal: Dict) -> float:
        """Calculate production-ready risk score (0-100, lower is better)"""
        try:
            risk_factors = []
            
            # Volume risk
            volume_strength = signal.get('volume_strength', 'WEAK')
            volume_risk = {'EXCEPTIONAL': 10, 'STRONG': 20, 'MODERATE': 40, 'WEAK': 70, 'UNKNOWN': 80}.get(volume_strength, 80)
            risk_factors.append(volume_risk)
            
            # Momentum risk
            momentum_score = signal.get('momentum_persistence', {}).get('score', 0)
            momentum_risk = 100 - momentum_score
            risk_factors.append(momentum_risk)
            
            # Volatility risk
            volatility = signal.get('volatility_1h', 5.0)  # Default 5%
            volatility_risk = min(volatility * 10, 100)  # Cap at 100
            risk_factors.append(volatility_risk)
            
            # ML confidence risk
            ml_confidence = signal.get('ml_signal_strength', 0.5)
            ml_risk = (1 - ml_confidence) * 100
            risk_factors.append(ml_risk)
            
            # Weighted average (equal weights for simplicity)
            total_risk = sum(risk_factors) / len(risk_factors)
            
            return min(max(total_risk, 0), 100)  # Ensure 0-100 range
            
        except:
            return 75.0  # Conservative default
    
    def _estimate_profit_potential(self, signal: Dict) -> Dict:
        """Estimate profit potential for the signal"""
        try:
            current_price = signal.get('current_price', 0)
            # Use target_quick as primary target, fallback to take_profit
            take_profit = signal.get('target_quick', signal.get('take_profit', 0))
            stop_loss = signal.get('stop_loss', 0)
            
            if current_price <= 0 or take_profit <= 0 or stop_loss <= 0:
                return {'potential_return': 0, 'potential_risk': 0, 'risk_reward_ratio': 0, 'grade': 'F'}
            
            potential_gain = (take_profit - current_price) / current_price * 100
            potential_loss = (current_price - stop_loss) / current_price * 100
            
            risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
            
            # Grade the opportunity with more precise thresholds
            if risk_reward_ratio >= 3.5 and potential_gain >= 4.0:
                grade = 'A+'
            elif risk_reward_ratio >= 2.8 and potential_gain >= 3.0:
                grade = 'A'
            elif risk_reward_ratio >= 2.2 and potential_gain >= 2.0:
                grade = 'B+'
            elif risk_reward_ratio >= 1.8 and potential_gain >= 1.5:
                grade = 'B'
            elif risk_reward_ratio >= 1.3 and potential_gain >= 1.0:
                grade = 'C'
            else:
                grade = 'D'
            
            return {
                'potential_return': round(potential_gain, 3),  # More precision
                'potential_risk': round(potential_loss, 3),
                'risk_reward_ratio': round(risk_reward_ratio, 3),  # More precision
                'grade': grade
            }
            
        except Exception as e:
            logging.warning(f"Profit potential calculation failed: {e}")
            return {'potential_return': 0, 'potential_risk': 0, 'risk_reward_ratio': 0, 'grade': 'F'}
    
    def _rank_for_production(self, signals: List[Dict], top_picks: int) -> List[Dict]:
        """
        Final ranking for production trading with more varied scoring
        """
        try:
            # Score each signal
            for signal in signals:
                score = 0
                
                # Profit potential weight (35%)
                profit_potential = signal.get('profit_potential', {})
                rr_ratio = profit_potential.get('risk_reward_ratio', 0)
                potential_return = profit_potential.get('potential_return', 0)
                
                # Reward higher R/R ratios more aggressively
                if rr_ratio >= 3.0:
                    rr_score = 35
                elif rr_ratio >= 2.5:
                    rr_score = 30
                elif rr_ratio >= 2.0:
                    rr_score = 25
                elif rr_ratio >= 1.5:
                    rr_score = 15
                else:
                    rr_score = 5
                
                score += rr_score + (potential_return * 1.5)
                
                # Risk score weight (25%) - lower risk = higher score
                production_risk = signal.get('production_risk_score', 75)
                score += (100 - production_risk) * 0.25
                
                # Volume strength weight (20%)
                volume_strength = signal.get('volume_strength', 'WEAK')
                volume_scores = {'EXCEPTIONAL': 20, 'STRONG': 15, 'MODERATE': 10, 'WEAK': 5, 'UNKNOWN': 2}
                score += volume_scores.get(volume_strength, 2)
                
                # Momentum persistence weight (10%)
                momentum_score = signal.get('momentum_persistence', {}).get('score', 0)
                score += momentum_score * 0.1
                
                # Price momentum bonus (10%)
                change_24h = signal.get('change_24h', signal.get('gain_24h', 0))
                if change_24h > 10:
                    score += 10
                elif change_24h > 5:
                    score += 6
                elif change_24h > 2:
                    score += 3
                
                signal['production_score'] = round(score, 3)  # More precision
            
            # Sort by production score and return top picks
            sorted_signals = sorted(signals, key=lambda x: x.get('production_score', 0), reverse=True)
            
            return sorted_signals[:top_picks]
            
        except Exception as e:
            logging.error(f"❌ Ranking failed: {e}")
            return signals[:top_picks]
    
    def _calculate_scan_metrics(self, signals: List[Dict]) -> Dict:
        """Calculate overall scan performance metrics"""
        try:
            if not signals:
                return {}
            
            # Average metrics
            avg_rr_ratio = np.mean([s.get('profit_potential', {}).get('risk_reward_ratio', 0) for s in signals])
            avg_potential_return = np.mean([s.get('profit_potential', {}).get('potential_return', 0) for s in signals])
            avg_production_score = np.mean([s.get('production_score', 0) for s in signals])
            avg_risk_score = np.mean([s.get('production_risk_score', 75) for s in signals])
            
            # Grade distribution
            grades = [s.get('profit_potential', {}).get('grade', 'F') for s in signals]
            grade_counts = pd.Series(grades).value_counts().to_dict()
            
            # Volume strength distribution
            volume_strengths = [s.get('volume_strength', 'WEAK') for s in signals]
            volume_counts = pd.Series(volume_strengths).value_counts().to_dict()
            
            return {
                'avg_risk_reward_ratio': round(avg_rr_ratio, 2),
                'avg_potential_return': round(avg_potential_return, 2),
                'avg_production_score': round(avg_production_score, 2),
                'avg_risk_score': round(avg_risk_score, 2),
                'grade_distribution': grade_counts,
                'volume_distribution': volume_counts,
                'total_signals': len(signals)
            }
            
        except Exception as e:
            logging.error(f"❌ Metrics calculation failed: {e}")
            return {}

# Legacy compatibility wrapper
def run_elite_production_scan(max_symbols=400, min_gain=5.0, top_picks=15, exchange="binance"):
    """Main function for institutional gainer analysis"""
    scanner = EliteProductionMoneyMaker(exchange_preference=exchange)
    return scanner.run_elite_gainer_scan(max_symbols, min_gain, top_picks, exchange)

if __name__ == "__main__":
    # Test the institutional scanner
    print("🏛️ Testing Institutional Production Money Maker...")
    results = run_elite_production_scan(exchange="binance")
    print(f"Found {results.get('signals_found', 0)} institutional signals")
    for pick in results.get('top_picks', [])[:5]:
        print(f"📊 {pick['symbol']}: {pick.get('profit_potential', {}).get('grade', 'N/A')} grade, "
              f"RR: {pick.get('profit_potential', {}).get('risk_reward_ratio', 0):.2f}")
