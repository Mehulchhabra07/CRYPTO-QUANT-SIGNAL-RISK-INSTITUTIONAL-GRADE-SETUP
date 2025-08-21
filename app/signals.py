import pandas as pd, numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import clean_ohlcv, ema, rsi, atr, boll_z, donchian, realized_vol_parkinson, composite_momentum, zscore
def add_indicators(df, cfg):
    df=clean_ohlcv(df).copy()
    if df.empty: return df
    f,s=cfg["signals"]["ma_fast"], cfg["signals"]["ma_slow"]
    df["EMA_fast"]=ema(df["Close"], f); df["EMA_slow"]=ema(df["Close"], s)
    df["RSI"]=rsi(df["Close"], cfg["signals"]["rsi_window"])
    df["ATR"]=atr(df,14); df["ATR_pct"]=100*df["ATR"]/(df["Close"]+1e-9)
    df["BOLLz"]=boll_z(df["Close"], cfg["signals"]["squeeze_window"])
    hi20,lo20=donchian(df, cfg["signals"]["donchian"])
    hi55,lo55=donchian(df, cfg["signals"]["donchian_long"])
    df["DonchianHi20"],df["DonchianLo20"]=hi20,lo20
    df["DonchianHi55"],df["DonchianLo55"]=hi55,lo55
    df["Mom"]=composite_momentum(df["Close"], cfg["signals"]["mom_windows"])
    df["RV_Parkinson"]=realized_vol_parkinson(df,20)
    return df
def multi_tf_indicators(dfs, cfg): return {k: add_indicators(d, cfg) for k,d in dfs.items()}
def trend_ok_tf(dfs_by_tf, rsi_floor):
    lo,hi=dfs_by_tf["lower"].dropna(), dfs_by_tf["higher"].dropna()
    if lo.empty or hi.empty: return False
    loL,hiL=lo.iloc[-1], hi.iloc[-1]
    return bool(loL["EMA_fast"]>loL["EMA_slow"] and loL["RSI"]>=rsi_floor and hiL["EMA_fast"]>hiL["EMA_slow"] and hiL["RSI"]>=rsi_floor)
def passes_risk_gates(dfs_by_tf, max_atr_pct, min_dollar_vol):
    lo=dfs_by_tf["lower"].dropna()
    if lo.empty: return (False, "no data")
    last=lo.iloc[-1]; 
    if np.isnan(last["ATR_pct"]) or last["ATR_pct"]>max_atr_pct: return (False, "atr too high")
    # Only check volume if min_dollar_vol > 0 (skip for backtest)
    if min_dollar_vol > 0:
        dv=float(last["Close"]*last.get("Volume",0.0))
        if dv<min_dollar_vol: return (False, "liquidity too low")
    return (True, "")
def score_asset_tf(dfs_by_tf, cfg):
    lo=dfs_by_tf["lower"].dropna()
    if lo.empty: return (-1e9, {})
    t=lo.tail(300)
    import numpy as np
    zmom=float(zscore(t["Mom"]).iloc[-1]) if "Mom" in t else 0.0
    ztrend=float(zscore(t["EMA_fast"]/(t["EMA_slow"]+1e-9)).iloc[-1]) if "EMA_fast" in t else 0.0
    zvol=float(zscore(-t["ATR_pct"]).iloc[-1]) if "ATR_pct" in t else 0.0
    w=cfg["scoring"]; total=float(w["w_momentum"]*zmom + w["w_trend"]*ztrend + w["w_volatility"]*zvol)
    last=lo.iloc[-1]; meta={"atr_pct": float(last["ATR_pct"]), "rsi": float(last["RSI"])}
    return total, meta
def strategy_trend_swing(dfs_by_tf, cfg):
    return (trend_ok_tf(dfs_by_tf, cfg["risk"]["rsi_floor"]), "ok")
def strategy_donchian_breakout(dfs_by_tf, cfg):
    lo=dfs_by_tf["lower"].dropna(); 
    if lo.empty: return (False, "no data")
    last=lo.iloc[-1]; cond= last["Close"]>last["DonchianHi20"]
    cond &= lo["DonchianHi55"].diff().tail(5).mean()>0
    cond &= last["BOLLz"]<2.0
    return (bool(cond), "ok" if cond else "no breakout")
def strategy_squeeze(dfs_by_tf, cfg):
    lo=dfs_by_tf["lower"].dropna(); 
    if lo.empty: return (False, "no data")
    squeeze= lo["RV_Parkinson"].rolling(20).mean().iloc[-1] < lo["RV_Parkinson"].rolling(100).mean().iloc[-1]
    aligned= lo.iloc[-1]["EMA_fast"]>lo.iloc[-1]["EMA_slow"]
    return (bool(squeeze and aligned), "ok" if squeeze and aligned else "no squeeze")
def regime_ok(df):
    if df is None or df.empty: return True
    last=df.dropna().iloc[-1]
    return bool(last["EMA_fast"]>last["EMA_slow"]) 
