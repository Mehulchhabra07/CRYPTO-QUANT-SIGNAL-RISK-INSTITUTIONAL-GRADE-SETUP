import numpy as np, pandas as pd
def clean_ohlcv(df):
    if df is None or df.empty: return pd.DataFrame()
    cols = {c:c.capitalize() for c in df.columns}
    df = df.rename(columns=cols)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open","High","Low","Close"]).sort_index()
def ema(x, span): return x.ewm(span=span, adjust=False).mean()
def rsi(x, w=14):
    d=x.diff(); up=d.clip(lower=0); dn=(-d).clip(lower=0)
    ru=up.rolling(w).mean(); rd=dn.rolling(w).mean(); rs=ru/(rd+1e-9); return 100-(100/(1+rs))
def atr(df, w=14):
    h,l,c=df["High"],df["Low"],df["Close"]; pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(w).mean()
def boll_z(c, w=20):
    m=c.rolling(w).mean(); s=c.rolling(w).std(ddof=0); return (c-m)/(s+1e-9)
def donchian(df, w=20):
    return df["High"].rolling(w).max(), df["Low"].rolling(w).min()
def realized_vol_parkinson(df, w=20):
    import numpy as np
    rng=(df["High"]/df["Low"]).apply(np.log); rv=(rng**2).rolling(w).mean()*(1/(4*np.log(2))); return (rv**0.5)
def composite_momentum(c, wins):
    import numpy as np
    s=0.0
    for w in wins:
        r=c.pct_change(w); s+= r/(r.rolling(10).std()+1e-9)
    return s
def zscore(s, w=100):
    m=s.rolling(w).mean(); sd=s.rolling(w).std(ddof=0); return (s-m)/(sd+1e-9)
