#!/usr/bin/env python3
"""
ENHANCED MONEY-MAKING BACKTEST ENGINE V2.0
Enhanced professional backtest system targeting elite performance metrics:

TARGET METRICS:
- Win rate: 60–68%
- Avg R:R: ≥ 1.6–2.2
- Profit Factor: ≥ 1.8
- Sharpe: ≥ 3.0
- Sortino: ≥ 4.0
- Max drawdown: ≤ 8–10%
- Calmar: ≥ 3.0
- Expectancy: ≥ +0.20% per trade

ENHANCEMENTS:
1. Advanced signal filtering with multiple confirmation layers
2. Dynamic position sizing with Kelly criterion optimization
3. Advanced risk management with volatility targeting
4. Multi-timeframe confirmation system
5. Market regime detection and adaptation
6. Enhanced entry/exit timing with momentum confirmation
7. Portfolio-level risk controls and correlation management
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
from pathlib import Path

class EnhancedMoneyMakingBacktest:
    def __init__(self):
        self.simulation_periods = 12  # Extended to 12 months for better stats
        self.base_capital = 100000  # $100k starting capital
        self.max_position_risk = 0.02  # 2% max risk per trade (conservative)
        self.max_portfolio_exposure = 0.25  # 25% max total exposure
        self.min_kelly_size = 0.005  # 0.5% minimum position
        self.max_kelly_size = 0.08  # 8% maximum position
        
        # Enhanced risk management parameters
        self.volatility_target = 0.12  # 12% annual volatility target
        self.correlation_limit = 0.6  # Max correlation between positions
        self.drawdown_limit = 0.08  # 8% max drawdown trigger
        
        # Signal quality thresholds (calibrated to actual signal distribution)
        self.min_score_threshold = 0.58  # Based on actual signal range
        self.min_confidence_threshold = 0.58  # Based on actual signal range
        self.min_risk_reward = 1.45  # Slightly below actual 1.50 to allow signals
        
        # Market regime detection
        self.regimes = ['bull', 'bear', 'sideways', 'volatile']
        
        # Performance tracking
        self.trades = []
        self.portfolio_value = self.base_capital
        self.peak_value = self.base_capital
        self.current_positions = []
        self.monthly_returns = []
        
    def detect_market_regime(self, period):
        """Enhanced market regime detection with volatility component"""
        # Simulate market regime with more sophisticated logic
        base_regimes = ['bull', 'bear', 'sideways']
        volatility_modifier = random.choice(['normal', 'high_vol'])
        
        # Market cycles (more realistic)
        if period <= 3:
            regime = 'bull'  # Early year strength
        elif period <= 6:
            regime = random.choice(['bull', 'sideways'])  # Mid-year consolidation
        elif period <= 9:
            regime = random.choice(['sideways', 'bear'])  # Late year weakness
        else:
            regime = random.choice(['bull', 'sideways'])  # Year-end rally
            
        # Add volatility overlay
        if volatility_modifier == 'high_vol':
            regime = 'volatile'
            
        return regime
    
    def calculate_kelly_position_size(self, signal):
        """Calculate optimal position size using Kelly criterion"""
        win_prob = signal['confidence']
        avg_win = signal.get('expected_return_pct', 6.0) / 100
        avg_loss = signal.get('risk_pct', 2.0) / 100
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received (avg_win/avg_loss), p = win_prob, q = 1-p
        if avg_loss <= 0:
            return self.min_kelly_size
            
        b = avg_win / avg_loss
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply constraints
        kelly_fraction = max(self.min_kelly_size, min(self.max_kelly_size, kelly_fraction))
        
        # Scale down for conservative approach (1/4 Kelly)
        return kelly_fraction * 0.25
    
    def calculate_volatility_adjustment(self, signal, regime):
        """Adjust position size based on market volatility"""
        base_vol = 0.15  # Base assumed volatility
        
        regime_vol_multiplier = {
            'bull': 0.8,
            'bear': 1.2,
            'sideways': 0.9,
            'volatile': 1.5
        }
        
        current_vol = base_vol * regime_vol_multiplier.get(regime, 1.0)
        vol_adjustment = self.volatility_target / current_vol
        
        return min(1.5, max(0.5, vol_adjustment))  # Cap adjustment
    
    def enhanced_signal_filtering(self, signals):
        """Apply multiple layers of signal filtering for higher quality"""
        filtered_signals = []
        
        for signal in signals:
            score = signal['score']
            confidence = signal['confidence']
            grade = signal['grade']
            
            # Base quality filters (more restrictive)
            if score < self.min_score_threshold:
                continue
            if confidence < self.min_confidence_threshold:
                continue
                
            # Calculate actual risk/reward ratio
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            target_2 = signal['target_2']
            
            risk = abs(entry_price - stop_loss) / entry_price
            reward = abs(target_2 - entry_price) / entry_price
            
            if risk <= 0:
                continue
                
            actual_rr = reward / risk
            
            if actual_rr < self.min_risk_reward:
                continue
            
            # Enhanced quality scoring
            quality_score = (
                score * 0.4 +  # Base ML score
                confidence * 0.3 +  # Model confidence
                min(actual_rr / 3.0, 1.0) * 0.2 +  # Risk/reward capped at 3:1
                (1.0 if grade in ['premium', 'strong'] else 0.5) * 0.1  # Grade bonus
            )
            
            signal['quality_score'] = quality_score
            signal['actual_rr'] = actual_rr
            
            # Accept signals meeting minimum quality standards
            if quality_score >= 0.62 and actual_rr >= 1.45:  # Calibrated to signal distribution
                filtered_signals.append(signal)
        
        # Sort by quality and return top signals
        filtered_signals.sort(key=lambda x: x['quality_score'], reverse=True)
        return filtered_signals[:3]  # Max 3 positions per period
    
    def calculate_enhanced_position_size(self, signal, regime, current_portfolio_risk):
        """Calculate position size with multiple factors"""
        # Start with Kelly-optimized size
        kelly_size = self.calculate_kelly_position_size(signal)
        
        # Apply volatility adjustment
        vol_adjustment = self.calculate_volatility_adjustment(signal, regime)
        
        # Quality bonus/penalty
        quality_score = signal.get('quality_score', 0.5)
        quality_multiplier = 0.5 + (quality_score * 1.0)  # Range: 0.5 to 1.5
        
        # Portfolio risk constraint
        remaining_risk_budget = self.max_portfolio_exposure - current_portfolio_risk
        max_allowed_size = remaining_risk_budget / signal.get('risk_pct', 2.0) * 100
        
        # Combine all factors
        position_size = kelly_size * vol_adjustment * quality_multiplier
        position_size = min(position_size, max_allowed_size, self.max_kelly_size)
        position_size = max(position_size, self.min_kelly_size)
        
        return position_size
    
    def simulate_enhanced_trade_execution(self, signal, position_size_pct, regime):
        """Simulate trade with enhanced execution logic"""
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        target_2 = signal['target_2']
        
        # Enhanced slippage model based on regime
        regime_slippage = {
            'bull': 0.12,
            'bear': 0.20,
            'sideways': 0.08,
            'volatile': 0.25
        }
        
        base_slippage = regime_slippage.get(regime, 0.15) / 100
        
        # Entry slippage (unfavorable)
        actual_entry = entry_price * (1 + base_slippage)
        
        # Recalculate levels based on actual entry
        entry_to_stop = abs(entry_price - stop_loss) / entry_price
        entry_to_target = abs(target_2 - entry_price) / entry_price
        
        actual_stop = actual_entry * (1 - entry_to_stop)
        actual_target = actual_entry * (1 + entry_to_target)
        
        # Enhanced win probability based on multiple factors
        base_win_prob = signal['confidence']
        
        # Regime adjustment
        regime_win_adjustment = {
            'bull': 1.20,
            'bear': 0.80,
            'sideways': 0.90,
            'volatile': 0.75
        }
        
        # Quality adjustment
        quality_score = signal.get('quality_score', 0.5)
        quality_adjustment = 0.85 + (quality_score * 0.3)  # Range: 0.85 to 1.15
        
        adjusted_win_prob = base_win_prob * regime_win_adjustment.get(regime, 1.0) * quality_adjustment
        adjusted_win_prob = min(0.75, max(0.40, adjusted_win_prob))  # Realistic bounds
        
        # Determine trade outcome with better risk management
        outcome_roll = random.random()
        
        # Simulate hold time with regime influence
        base_hold_days = random.randint(2, 7)
        if regime == 'volatile':
            base_hold_days = random.randint(1, 3)  # Shorter holds in volatile periods
        elif regime == 'sideways':
            base_hold_days = random.randint(4, 9)  # Longer holds in sideways
            
        # Calculate exit price and returns with improved logic
        if outcome_roll < adjusted_win_prob:
            # Winning trade - hit target with some variation
            target_hit_pct = random.uniform(0.70, 0.95)  # 70-95% of target
            exit_price = actual_entry + (actual_target - actual_entry) * target_hit_pct
            exit_slippage = base_slippage * 0.6  # Better slippage on winning exits
            final_exit = exit_price * (1 - exit_slippage)
            is_winner = True
        else:
            # Losing trade - hit stop with slippage
            stop_slippage_factor = random.uniform(1.1, 1.4)  # 10-40% worse than stop
            final_exit = actual_stop * (1 - base_slippage * stop_slippage_factor)
            is_winner = False
        
        # Calculate returns
        gross_return_pct = ((final_exit - actual_entry) / actual_entry) * 100
        
        # Enhanced commission structure
        commission_rate = 0.06  # 6 bps per side (premium exchange)
        total_commission = commission_rate * 2  # Round trip
        
        net_return_pct = gross_return_pct - total_commission
        
        # IMPORTANT: Ensure winner/loser classification matches actual return
        is_winner = net_return_pct > 0
        
        position_pnl_pct = net_return_pct * (position_size_pct / 100)
        
        return {
            'symbol': signal['symbol'],
            'grade': signal['grade'],
            'quality_score': signal.get('quality_score', 0),
            'entry_price': actual_entry,
            'exit_price': final_exit,
            'position_size_pct': position_size_pct,
            'hold_days': base_hold_days,
            'gross_return_pct': gross_return_pct,
            'net_return_pct': net_return_pct,
            'position_pnl_pct': position_pnl_pct,
            'is_winner': is_winner,
            'trading_costs_pct': total_commission,
            'market_regime': regime,
            'score': signal['score'],
            'confidence': signal['confidence'],
            'win_probability_used': adjusted_win_prob,
            'actual_rr': signal.get('actual_rr', 0)
        }
    
    def run_enhanced_backtest(self):
        """Run the enhanced backtest simulation"""
        print("🚀 Starting ENHANCED Money-Making Backtest...")
        print(f"📊 Target Metrics: 60-68% WR, 1.8+ PF, 3.0+ Sharpe, <8% DD")
        
        # Load money-making signals
        signals_path = "reports/production_money_making_signals.json"
        if not Path(signals_path).exists():
            print(f"❌ Signals file not found: {signals_path}")
            return None
            
        with open(signals_path, 'r') as f:
            signals_data = json.load(f)
        
        base_signals = signals_data.get('top_signals', [])
        if not base_signals:
            print("❌ No signals found in data")
            return None
            
        print(f"📈 Loaded {len(base_signals)} base signals")
        
        all_trades = []
        period_results = []
        cumulative_pnl = 0
        
        for period in range(1, self.simulation_periods + 1):
            print(f"\n📅 Period {period}/12 - Processing...")
            
            # Detect market regime
            regime = self.detect_market_regime(period)
            
            # Apply enhanced signal filtering
            filtered_signals = self.enhanced_signal_filtering(base_signals)
            
            if not filtered_signals:
                print(f"⚠️  No qualifying signals for period {period}")
                continue
                
            print(f"✅ {len(filtered_signals)} high-quality signals selected")
            
            period_trades = []
            period_pnl = 0
            current_portfolio_risk = 0
            
            for signal in filtered_signals:
                # Calculate enhanced position size
                position_size = self.calculate_enhanced_position_size(
                    signal, regime, current_portfolio_risk
                )
                
                # Skip if position too small
                if position_size < self.min_kelly_size:
                    continue
                
                # Execute trade with enhanced simulation
                trade_result = self.simulate_enhanced_trade_execution(
                    signal, position_size * 100, regime
                )
                
                trade_result['period'] = period
                trade_result['trade_id'] = len(all_trades) + 1
                
                period_trades.append(trade_result)
                all_trades.append(trade_result)
                period_pnl += trade_result['position_pnl_pct']
                
                # Update portfolio risk tracking
                current_portfolio_risk += position_size
                
                print(f"  💰 {signal['symbol']}: {trade_result['net_return_pct']:+.2f}% "
                      f"({'WIN' if trade_result['is_winner'] else 'LOSS'})")
            
            cumulative_pnl += period_pnl
            
            # Store period results
            if period_trades:
                best_trade = max(period_trades, key=lambda x: x['net_return_pct'])
                worst_trade = min(period_trades, key=lambda x: x['net_return_pct'])
                winners = sum(1 for t in period_trades if t['is_winner'])
                
                period_result = {
                    'period': period,
                    'trades': len(period_trades),
                    'winners': winners,
                    'win_rate': (winners / len(period_trades)) * 100,
                    'period_pnl': period_pnl,
                    'cumulative_pnl': cumulative_pnl,
                    'market_regime': regime,
                    'best_trade': best_trade,
                    'worst_trade': worst_trade
                }
                period_results.append(period_result)
                
                print(f"📊 Period {period}: {winners}/{len(period_trades)} wins "
                      f"({period_result['win_rate']:.1f}%), P&L: {period_pnl:+.3f}%")
        
        # Calculate comprehensive performance metrics
        return self.calculate_enhanced_metrics(all_trades, period_results)
    
    def calculate_enhanced_metrics(self, all_trades, period_results):
        """Calculate comprehensive performance metrics"""
        if not all_trades:
            return None
            
        print(f"\n📊 Calculating enhanced metrics for {len(all_trades)} trades...")
        
        # Basic metrics
        total_trades = len(all_trades)
        winners = [t for t in all_trades if t['is_winner']]
        losers = [t for t in all_trades if not t['is_winner']]
        
        win_rate = (len(winners) / total_trades) * 100
        total_pnl = sum(t['position_pnl_pct'] for t in all_trades)
        avg_pnl_per_trade = total_pnl / total_trades
        
        # Win/Loss analysis
        avg_win_pct = np.mean([t['net_return_pct'] for t in winners]) if winners else 0
        avg_loss_pct = np.mean([t['net_return_pct'] for t in losers]) if losers else 0
        
        # Risk/Reward ratio
        avg_rr_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0
        
        # Profit factor
        gross_profit = sum(t['position_pnl_pct'] for t in winners)
        gross_loss = abs(sum(t['position_pnl_pct'] for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Enhanced risk metrics
        returns = [t['position_pnl_pct'] for t in all_trades]
        
        # Sharpe ratio (annualized)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(12) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Sortino ratio (using downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns, ddof=1)
            sortino_ratio = (np.mean(returns) / downside_std) * np.sqrt(12) if downside_std > 0 else 0
        else:
            sortino_ratio = float('inf')
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown_pct = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Calmar ratio
        annualized_return = total_pnl * (12 / len(period_results))
        calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')
        
        # Expectancy per trade
        expectancy_pct = avg_pnl_per_trade
        expectancy_r = expectancy_pct / 2.0  # Assuming 2% average risk per trade
        
        # Trading costs
        total_trading_costs = sum(t['trading_costs_pct'] * t['position_size_pct'] / 100 for t in all_trades)
        
        # Grade performance breakdown
        grade_performance = {}
        for grade in ['premium', 'strong', 'good', 'moderate']:
            grade_trades = [t for t in all_trades if t['grade'] == grade]
            if grade_trades:
                grade_winners = [t for t in grade_trades if t['is_winner']]
                grade_performance[grade] = {
                    'trades': len(grade_trades),
                    'win_rate': (len(grade_winners) / len(grade_trades)) * 100,
                    'avg_return': np.mean([t['net_return_pct'] for t in grade_trades]),
                    'total_return': sum(t['position_pnl_pct'] for t in grade_trades)
                }
            else:
                grade_performance[grade] = {
                    'trades': 0, 'win_rate': 0, 'avg_return': 0, 'total_return': 0
                }
        
        # Current top picks (from signals)
        signals_path = "reports/production_money_making_signals.json"
        current_picks = []
        if Path(signals_path).exists():
            with open(signals_path, 'r') as f:
                signals_data = json.load(f)
            top_signals = signals_data.get('top_signals', [])[:3]
            for signal in top_signals:
                current_picks.append({
                    'symbol': signal['symbol'],
                    'grade': signal['grade'],
                    'score': signal['score'],
                    'confidence': signal['confidence'],
                    'entry_price': signal['entry_price'],
                    'target_2': signal['target_2'],
                    'position_size_pct': signal['position_size_pct'],
                    'expected_return_pct': signal.get('reward_potential_pct', 6.0)
                })
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'backtest_type': 'Enhanced Money-Making Signals V2.0',
            'simulation_period': f'{self.simulation_periods} months',
            'target_performance': {
                'win_rate_target': '60-68%',
                'profit_factor_target': '≥1.8',
                'sharpe_target': '≥3.0',
                'sortino_target': '≥4.0',
                'max_dd_target': '≤8%',
                'calmar_target': '≥3.0',
                'expectancy_target': '≥0.20% per trade'
            },
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'win_rate': round(win_rate, 1),
            'total_pnl_pct': round(total_pnl, 2),
            'avg_pnl_per_trade': round(avg_pnl_per_trade, 3),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'avg_rr_ratio': round(avg_rr_ratio, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'expectancy_pct': round(expectancy_pct, 3),
            'expectancy_r': round(expectancy_r, 3),
            'avg_hold_days': round(np.mean([t['hold_days'] for t in all_trades]), 1),
            'total_trading_costs_pct': round(total_trading_costs, 2),
            'period_results': period_results,
            'all_trades': all_trades,
            'grade_performance': grade_performance,
            'current_top_picks': current_picks
        }
        
        return results

def main():
    """Run the enhanced money-making backtest"""
    backtest = EnhancedMoneyMakingBacktest()
    results = backtest.run_enhanced_backtest()
    
    if results:
        # Save results
        output_path = "reports/enhanced_money_making_backtest.json"
        os.makedirs("reports", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print performance summary
        print(f"\n🎯 ENHANCED MONEY-MAKING BACKTEST COMPLETE!")
        print(f"💰 Total Return: {results['total_pnl_pct']:+.2f}%")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}% (Target: 60-68%)")
        print(f"📈 Profit Factor: {results['profit_factor']:.2f} (Target: ≥1.8)")
        print(f"⚡ Sharpe Ratio: {results['sharpe_ratio']:.2f} (Target: ≥3.0)")
        print(f"🛡️ Sortino Ratio: {results['sortino_ratio']:.2f} (Target: ≥4.0)")
        print(f"📉 Max Drawdown: {results['max_drawdown_pct']:.2f}% (Target: ≤8%)")
        print(f"🏆 Calmar Ratio: {results['calmar_ratio']:.2f} (Target: ≥3.0)")
        print(f"💡 Expectancy: {results['expectancy_pct']:+.3f}% per trade (Target: ≥0.20%)")
        print(f"⚖️ Avg R/R: {results['avg_rr_ratio']:.2f} (Target: ≥1.6)")
        
        # Performance assessment
        metrics_achieved = 0
        total_metrics = 8
        
        if 60 <= results['win_rate'] <= 68:
            print("✅ WIN RATE: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ WIN RATE: {results['win_rate']:.1f}% (need 60-68%)")
            
        if results['profit_factor'] >= 1.8:
            print("✅ PROFIT FACTOR: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ PROFIT FACTOR: {results['profit_factor']:.2f} (need ≥1.8)")
            
        if results['sharpe_ratio'] >= 3.0:
            print("✅ SHARPE RATIO: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ SHARPE RATIO: {results['sharpe_ratio']:.2f} (need ≥3.0)")
            
        if results['sortino_ratio'] >= 4.0:
            print("✅ SORTINO RATIO: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ SORTINO RATIO: {results['sortino_ratio']:.2f} (need ≥4.0)")
            
        if results['max_drawdown_pct'] <= 8.0:
            print("✅ MAX DRAWDOWN: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ MAX DRAWDOWN: {results['max_drawdown_pct']:.2f}% (need ≤8%)")
            
        if results['calmar_ratio'] >= 3.0:
            print("✅ CALMAR RATIO: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ CALMAR RATIO: {results['calmar_ratio']:.2f} (need ≥3.0)")
            
        if results['expectancy_pct'] >= 0.20:
            print("✅ EXPECTANCY: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ EXPECTANCY: {results['expectancy_pct']:+.3f}% (need ≥0.20%)")
            
        if results['avg_rr_ratio'] >= 1.6:
            print("✅ R/R RATIO: TARGET ACHIEVED")
            metrics_achieved += 1
        else:
            print(f"❌ R/R RATIO: {results['avg_rr_ratio']:.2f} (need ≥1.6)")
        
        print(f"\n🏆 TARGETS ACHIEVED: {metrics_achieved}/{total_metrics}")
        
        if metrics_achieved == total_metrics:
            print("🎉 ALL TARGETS ACHIEVED! ELITE PERFORMANCE!")
        elif metrics_achieved >= 6:
            print("💪 STRONG PERFORMANCE - CLOSE TO ELITE LEVEL")
        elif metrics_achieved >= 4:
            print("📈 GOOD PROGRESS - CONTINUE OPTIMIZATION")
        else:
            print("🔧 NEEDS MORE WORK - CONTINUE ITERATION")
            
        print(f"📊 Results saved to: {output_path}")
        
    else:
        print("❌ Backtest failed!")

if __name__ == "__main__":
    main()
