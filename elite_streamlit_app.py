import json, streamlit as st, pandas as pd, time, os
from pathlib import Path
import datetime
import sys

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

st.set_page_config(page_title="MEHUL Quant Risk — Institutional Grade Setup", layout="wide")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Institutional Scanner", "📈 Live Results", "🎯 Top Picks", "🔬 Strategy Analysis"])

with tab1:
    # Main header
    st.title("🏛️ MEHUL QUANT RISK — INSTITUTIONAL GRADE SETUP 🏛️")
    st.markdown("**⚡ INSTITUTIONAL-GRADE ANALYTICS + ML VALIDATION + RISK MANAGEMENT = PROFESSIONAL PROFITS**")
    st.caption("🎯 **Mission**: Professional-grade quantitative analysis with institutional risk management and ML validation")
    
    # Institutional features box
    st.success("""
    **🏛️ INSTITUTIONAL GRADE FEATURES:**
    - 📊 **PROFESSIONAL ANALYTICS**: Advanced quantitative models and risk metrics
    - 🧠 **ML VALIDATION**: 51-model ensemble for institutional-grade signal validation  
    - 🎯 **RISK MANAGEMENT**: Institutional-level position sizing and portfolio optimization
    - ⚡ **MULTI-EXCHANGE**: Binance, Kraken and other premium exchanges
    - 🛡️ **COMPLIANCE READY**: Professional-grade reporting and audit trails
    - 💎 **INSTITUTIONAL SETUP**: Real institutional money management protocols
    """)
    
    # Configuration section
    st.subheader("⚙️ Institutional Scanner Configuration")
    
    config_col1, config_col2, config_col3, config_col4 = st.columns(4)
    
    with config_col1:
        exchange_option = st.selectbox("Exchange", 
                                     options=["Binance (All Markets)", "Kraken Only", "Multi-Exchange"],
                                     index=0,
                                     help="Select which exchange to analyze")
        
    with config_col2:
        max_symbols = st.number_input("Max Symbols to Scan", min_value=50, max_value=1000, value=400, step=50, 
                                     help="Number of symbols to scan for opportunities")
        
    with config_col3:
        min_gain_24h = st.number_input("Min 24h Gain %", min_value=0.5, max_value=50.0, value=3.0, step=0.5,
                                      help="Minimum 24h gain percentage to consider")
        
    with config_col4:
        top_picks = st.number_input("Top Picks", min_value=5, max_value=50, value=15, step=5,
                                   help="Number of top picks to return")
    
    # Main scan button section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🏛️ RUN INSTITUTIONAL SCAN", help="Professional institutional-grade market analysis!", type="primary", use_container_width=True):
            with st.spinner("🔍 Running institutional analysis... This may take a few minutes..."):
                # Show progress info
                progress_info = st.empty()
                exchange_text = str(exchange_option).split(" ")[0] if exchange_option else "Binance"
                progress_info.info(f"🔄 Scanning {max_symbols} symbols on {exchange_text} with institutional-grade analytics...")
                
                try:
                    # Import and run the elite scanner
                    from elite_production_money_maker import run_elite_production_scan
                    
                    # Determine exchange parameter
                    exchange_param = "kraken" if "Kraken" in str(exchange_option) else "binance"
                    
                    # Run the institutional scan
                    results = run_elite_production_scan(
                        max_symbols=max_symbols,
                        min_gain=min_gain_24h,
                        top_picks=top_picks,
                        exchange=exchange_param
                    )
                    
                    progress_info.empty()
                    
                    if results and 'top_picks' in results and results['signals_found'] > 0:
                        st.success(f"✅ Found {results['signals_found']} institutional-grade signals on {exchange_text}!")
                        
                        # Display results
                        st.subheader("�️ INSTITUTIONAL SIGNALS")
                        
                        # Create DataFrame for display
                        signals_data = []
                        for signal in results['top_picks']:
                            # Get the correct target price
                            target_price = signal.get('target_quick', signal.get('take_profit', 0))
                            
                            signals_data.append({
                                'Symbol': signal.get('symbol', 'N/A'),
                                'Grade': signal.get('profit_potential', {}).get('grade', 'N/A'),
                                'Current Price': f"${signal.get('current_price', 0):.6f}",
                                'Stop Loss': f"${signal.get('stop_loss', 0):.6f}",
                                'Target Price': f"${target_price:.6f}",
                                'Risk/Reward': f"{signal.get('profit_potential', {}).get('risk_reward_ratio', 0):.3f}",
                                'Potential %': f"{signal.get('profit_potential', {}).get('potential_return', 0):.2f}%",
                                'Risk %': f"{signal.get('profit_potential', {}).get('potential_risk', 0):.2f}%",
                                'Volume Strength': signal.get('volume_strength', 'N/A'),
                                'Institutional Score': f"{signal.get('production_score', 0):.2f}"
                            })
                        
                        if signals_data:
                            df = pd.DataFrame(signals_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Performance metrics
                            if 'performance_metrics' in results:
                                metrics = results['performance_metrics']
                                
                                st.subheader("📊 Scan Performance Metrics")
                                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                
                                with metric_col1:
                                    st.metric("Avg R/R Ratio", f"{metrics.get('avg_risk_reward_ratio', 0):.2f}")
                                    
                                with metric_col2:
                                    st.metric("Avg Potential Return", f"{metrics.get('avg_potential_return', 0):.2f}%")
                                    
                                with metric_col3:
                                    st.metric("Avg Production Score", f"{metrics.get('avg_production_score', 0):.1f}")
                                    
                                with metric_col4:
                                    st.metric("Total Signals", metrics.get('total_signals', 0))
                                    
                            # Save results for other tabs
                            st.session_state['latest_results'] = results
                            
                    else:
                        st.warning("🔍 No elite gainer signals found with current criteria. Try adjusting parameters.")
                        
                except Exception as e:
                    progress_info.empty()
                    st.error(f"❌ Error running elite gainer scan: {str(e)}")
                    st.code(str(e), language="text")
    
    with col2:
        if st.button("📊 Quick Scan (100 symbols)", help="Faster scan with top 100 symbols", use_container_width=True):
            with st.spinner("⚡ Quick scanning top 100 symbols..."):
                try:
                    from elite_production_money_maker import run_elite_production_scan
                    
                    results = run_elite_production_scan(
                        max_symbols=100,
                        min_gain=3.0,
                        top_picks=10
                    )
                    
                    if results and 'signals_found' in results and results['signals_found'] > 0:
                        st.success(f"⚡ Quick scan found {results['signals_found']} signals!")
                        st.session_state['latest_results'] = results
                        st.rerun()
                    else:
                        st.info("⚡ Quick scan completed - no signals found")
                        
                except Exception as e:
                    st.error(f"❌ Quick scan error: {str(e)}")

with tab2:
    st.subheader("📈 Live Scanning Results")
    
    if 'latest_results' in st.session_state:
        results = st.session_state['latest_results']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Signals Found", results.get('signals_found', 0))
            
        with col2:
            st.metric("Total Scanned", results.get('total_scanned', 0))
            
        with col3:
            scan_time = results.get('scan_time', '')
            if scan_time:
                try:
                    dt = datetime.datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
                    st.metric("Last Scan", dt.strftime("%H:%M:%S"))
                except:
                    st.metric("Last Scan", "Unknown")
            else:
                st.metric("Last Scan", "Never")
                
        with col4:
            if 'performance_metrics' in results:
                avg_rr = results['performance_metrics'].get('avg_risk_reward_ratio', 0)
                st.metric("Avg R/R", f"{avg_rr:.2f}")
            else:
                st.metric("Avg R/R", "N/A")
        
        # Detailed results
        if 'top_picks' in results and results['top_picks']:
            st.subheader("📋 Signal Details")
            
            for i, signal in enumerate(results['top_picks'][:10], 1):
                profit_potential = signal.get('profit_potential', {})
                target_price = signal.get('target_quick', signal.get('take_profit', 0))
                
                with st.expander(f"{i}. {signal.get('symbol', 'N/A')} - Grade: {profit_potential.get('grade', 'N/A')} - R/R: {profit_potential.get('risk_reward_ratio', 0):.3f}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Price Info:**")
                        st.write(f"Current: ${signal.get('current_price', 0):.6f}")
                        st.write(f"Stop Loss: ${signal.get('stop_loss', 0):.6f}")
                        st.write(f"Target: ${target_price:.6f}")
                        
                    with col2:
                        st.write("**Performance:**")
                        change_24h = signal.get('change_24h', signal.get('gain_24h', 0))
                        change_1h = signal.get('change_1h', signal.get('gain_1h', 0))
                        st.write(f"24h Change: {change_24h:.2f}%")
                        st.write(f"1h Change: {change_1h:.2f}%")
                        st.write(f"Volume: {signal.get('volume_strength', 'N/A')}")
                        
                    with col3:
                        st.write("**Risk/Reward:**")
                        st.write(f"Potential Return: {profit_potential.get('potential_return', 0):.2f}%")
                        st.write(f"Potential Risk: {profit_potential.get('potential_risk', 0):.2f}%")
                        st.write(f"Risk/Reward: {profit_potential.get('risk_reward_ratio', 0):.3f}")
                        st.write(f"Production Score: {signal.get('production_score', 0):.2f}")
        
    else:
        st.info("👈 Run a scan in the first tab to see results here")

with tab3:
    st.subheader("🎯 Institutional Analysis")
    
    if 'latest_results' in st.session_state and 'top_picks' in st.session_state['latest_results']:
        top_picks = st.session_state['latest_results']['top_picks']
        
        if top_picks:
            # Top 5 picks
            st.write("**�️ TOP 5 INSTITUTIONAL PICKS:**")
            
            for i, pick in enumerate(top_picks[:5], 1):
                profit_potential = pick.get('profit_potential', {})
                grade = profit_potential.get('grade', 'N/A')
                target_price = pick.get('target_quick', pick.get('take_profit', 0))
                
                # Color based on grade
                if grade in ['A+', 'A']:
                    color = "🟢"
                elif grade in ['B+', 'B']:
                    color = "🟡"
                else:
                    color = "🔴"
                
                change_24h = pick.get('change_24h', pick.get('gain_24h', 0))
                
                st.markdown(f"""
                **{color} #{i} - {pick.get('symbol', 'N/A')}** (Grade: {grade})
                - 💰 Potential Return: {profit_potential.get('potential_return', 0):.2f}%
                - 🛡️ Risk/Reward: {profit_potential.get('risk_reward_ratio', 0):.3f}
                - 📊 24h Change: {change_24h:.2f}%
                - 💲 Current: ${pick.get('current_price', 0):.6f} → Target: ${target_price:.6f}
                - 🔥 Volume: {pick.get('volume_strength', 'N/A')}
                - ⭐ Score: {pick.get('production_score', 0):.2f}
                """)
                st.markdown("---")
            
            # Performance distribution
            st.subheader("📊 Performance Distribution")
            
            if len(top_picks) > 1:
                # Grade distribution
                grades = [p.get('profit_potential', {}).get('grade', 'F') for p in top_picks]
                grade_counts = pd.Series(grades).value_counts()
                
                if not grade_counts.empty:
                    st.bar_chart(grade_counts)
                
                # Volume strength distribution
                volume_strengths = [p.get('volume_strength', 'WEAK') for p in top_picks]
                volume_counts = pd.Series(volume_strengths).value_counts()
                
                if not volume_counts.empty:
                    st.bar_chart(volume_counts)
            
    else:
        st.info("👈 Run a scan to see top picks analysis")

with tab4:
    st.subheader("🔬 Strategy Analysis - How Our System Works")
    
    st.markdown("""
    ## 🏛️ **MEHUL Quant Risk Institutional Analysis Framework**
    
    Our institutional-grade system employs a comprehensive multi-layer analysis approach to identify the highest quality trading opportunities. Here's exactly how we analyze each symbol:
    """)
    
    # Phase 1: Data Collection
    st.markdown("### 📊 **Phase 1: Multi-Exchange Data Collection**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔗 Exchange Integration:**
        - **Binance**: 400+ USDT pairs with real-time data
        - **Kraken**: Institutional-grade USD pairs (BTC/USD, ETH/USD, etc.)
        - **Volume Filtering**: Minimum $100k daily volume requirement
        - **Liquidity Check**: Active markets with tight spreads only
        
        **📈 Market Data Collected:**
        - Current price and 24h price changes
        - Volume data (24h, 7-day average)
        - OHLCV data across multiple timeframes (5m, 1h, 1d)
        - Orderbook depth and spread analysis
        """)
    
    with col2:
        st.markdown("""
        **⏱️ Timeframe Analysis:**
        - **5-minute**: Short-term momentum and volatility
        - **1-hour**: Intraday trend confirmation
        - **1-day**: Medium-term trend analysis
        
        **🎯 Symbol Universe:**
        - **Binance**: All USDT pairs with >$100k volume
        - **Kraken**: USD pairs for institutional compliance
        - **Quality Filter**: Active, liquid markets only
        - **Real-time**: Live market data every scan
        """)
    
    # Phase 2: Technical Analysis
    st.markdown("### 📐 **Phase 2: Advanced Technical Analysis**")
    
    st.markdown("""
    **🔢 Technical Indicators Calculated:**
    
    1. **Moving Averages:**
       - SMA 20 & 50 (trend direction)
       - EMA 12 & 26 (momentum)
       - Price position relative to MAs
    
    2. **MACD (Moving Average Convergence Divergence):**
       - MACD line (EMA12 - EMA26)
       - Signal line (9-period EMA of MACD)
       - Histogram (MACD - Signal)
       - Bullish/bearish crossovers
    
    3. **RSI (Relative Strength Index):**
       - 14-period RSI calculation
       - Overbought/oversold levels (30/70)
       - Momentum strength assessment
       - Divergence detection
    
    4. **Volume Analysis:**
       - Volume vs 20-day average ratio
       - Volume confirmation for price moves
       - Institutional volume thresholds
    
    5. **Bollinger Bands:**
       - 20-period SMA with 2 standard deviations
       - Price position within bands
       - Volatility and mean reversion signals
    
    6. **Price Action:**
       - Support and resistance levels
       - Recent high/low analysis
       - Breakout pattern recognition
    """)
    
    # Phase 3: ML Validation
    st.markdown("### 🧠 **Phase 3: Machine Learning Validation (51 Models)**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Model Ensemble:**
        - **51 different ML models** trained on historical data
        - **Random Forest, XGBoost, Neural Networks**
        - **Feature Engineering**: 100+ technical features
        - **Cross-validation**: Multiple timeframe validation
        
        **📊 Feature Categories:**
        - **Price Features**: Returns, volatility, momentum
        - **Volume Features**: Volume ratios, patterns
        - **Technical Features**: All indicators mentioned above
        - **Market Structure**: Support/resistance, trends
        """)
    
    with col2:
        st.markdown("""
        **✅ ML Validation Process:**
        - Each signal scored by all 51 models
        - **Confidence Score**: Percentage of models agreeing
        - **Signal Strength**: Weighted average prediction
        - **Risk Assessment**: Model uncertainty analysis
        
        **🎯 Model Performance:**
        - Trained on 2+ years of market data
        - Backtested across multiple market conditions
        - Continuously updated with new data
        """)
    
    # Phase 4: Risk Management
    st.markdown("### 🛡️ **Phase 4: Institutional Risk Management**")
    
    st.markdown("""
    **💰 Position Sizing & Risk Controls:**
    
    1. **Adaptive Stop Loss Calculation:**
       - Based on ATR (Average True Range) - 14 periods
       - Volatility multiplier: 1.5x for high confidence, 2.0x for moderate
       - Dynamic adjustment based on market conditions
       - Maximum stop loss: 2.5% for day trading focus
    
    2. **Target Price Determination:**
       - **Quick Target**: 2-3% for fast profits
       - **Stretch Target**: 4-6% for longer holds
       - Risk/Reward ratio minimum: 1.5:1
       - Based on recent volatility and support/resistance
    
    3. **Portfolio Risk Management:**
       - Maximum 1% total portfolio risk per trade
       - Position sizing based on individual stop loss
       - Correlation analysis to avoid concentration
       - Maximum 15% position size per symbol
    
    4. **Volume & Liquidity Verification:**
       - Minimum $100k daily volume
       - Spread analysis for execution quality
       - Market depth verification
       - Institutional liquidity standards
    """)
    
    # Phase 5: Scoring & Ranking
    st.markdown("### 🏆 **Phase 5: Advanced Scoring & Ranking System**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Production Score Components:**
        
        **1. Profit Potential (35%):**
        - Risk/Reward ratio weighting
        - Potential return percentage
        - Target achievement probability
        
        **2. Risk Assessment (25%):**
        - Volatility analysis
        - Market structure quality
        - Liquidity and execution risk
        """)
    
    with col2:
        st.markdown("""
        **3. Volume Confirmation (20%):**
        - EXCEPTIONAL: >3x average volume
        - STRONG: >2x average volume  
        - MODERATE: >1.5x average volume
        
        **4. Momentum Persistence (20%):**
        - Multi-timeframe alignment
        - Trend continuation probability
        - Market sentiment analysis
        """)
    
    # Grading System
    st.markdown("### 🎯 **Grade Assignment System**")
    
    grade_col1, grade_col2, grade_col3 = st.columns(3)
    
    with grade_col1:
        st.markdown("""
        **🟢 A+ Grade:**
        - Risk/Reward ≥ 3.5
        - Potential Return ≥ 4%
        - High ML confidence (>80%)
        - Exceptional volume
        """)
    
    with grade_col2:
        st.markdown("""
        **🟡 B+ Grade:**
        - Risk/Reward ≥ 2.2
        - Potential Return ≥ 2%
        - Good ML confidence (>65%)
        - Strong volume confirmation
        """)
    
    with grade_col3:
        st.markdown("""
        **🔴 C Grade:**
        - Risk/Reward ≥ 1.3
        - Potential Return ≥ 1%
        - Moderate confidence (>50%)
        - Basic volume requirements
        """)
    
    # Final Selection
    st.markdown("### ✅ **Final Signal Selection Process**")
    
    st.markdown("""
    **🔍 Multi-Layer Filtering:**
    
    1. **Technical Filter**: Must pass all technical criteria
    2. **ML Filter**: Minimum 50% model agreement (adjustable)
    3. **Risk Filter**: Risk/Reward ratio ≥ 1.5
    4. **Volume Filter**: Institutional volume standards
    5. **Liquidity Filter**: Execution quality verification
    6. **Final Ranking**: By production score (highest first)
    
    **📈 Output Format:**
    - Symbol with exchange identification
    - Entry price (current market price)
    - Stop loss (risk management level)
    - Target price (profit objective)
    - Risk/Reward ratio (profit potential)
    - Grade (A+ to D quality rating)
    - Production score (overall ranking)
    - Volume strength assessment
    - Institutional compliance status
    """)
    
    # Performance Metrics
    st.markdown("### 📊 **Performance Tracking & Compliance**")
    
    st.markdown("""
    **🏛️ Institutional Standards:**
    - **Audit Trail**: Complete analysis methodology documentation
    - **Risk Metrics**: Portfolio-level risk monitoring
    - **Performance Attribution**: Strategy component analysis
    - **Compliance Reporting**: Regulatory-ready documentation
    - **Real-time Monitoring**: Live market condition adaptation
    
    **📈 Continuous Improvement:**
    - **Model Retraining**: Weekly model updates
    - **Strategy Backtesting**: Historical performance validation
    - **Market Regime Detection**: Adaptive strategy parameters
    - **Risk Model Calibration**: Ongoing risk metric refinement
    """)
    
    st.success("""
    💡 **Key Advantage**: Our system combines traditional technical analysis with cutting-edge machine learning 
    and institutional-grade risk management to identify the highest probability opportunities while maintaining 
    strict risk controls suitable for professional money management.
    """)

# Footer
st.markdown("---")
st.markdown("🏛️ **MEHUL Quant Risk — Institutional Grade Setup** - Professional quantitative analysis for serious institutional investors")
st.caption("⚠️ **Risk Warning**: Institutional-grade trading involves substantial risk. Only trade with proper risk management and compliance protocols.")
