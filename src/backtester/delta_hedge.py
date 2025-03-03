#!/usr/bin/env python3
from datetime import timedelta
import numpy as np, pandas as pd, matplotlib.pyplot as plt, json
from option_calculator import OptionCalculator
from math import sqrt

SHARES_TO_HEDGE=10_000
DTE_MIN,DTE_MAX=10,45
RISK_FREE_RATE=0.02
TRADING_DAYS=252
OPT_MULT=100
MAX_CONTRACTS=3_000
DATA_CSV="data/spy_options.csv"
SLIPPAGE_PERC=0.001
COMMISSION_PER_CONTRACT=0.65

def load_options(path):
    cols=["date","exdate","cp_flag","strike_price","best_bid",
          "best_offer","impl_volatility"]
    df=pd.read_csv(path,usecols=cols,parse_dates=["date","exdate"])
    df["strike_price"]/=1_000; df.set_index("date",inplace=True)
    return df.sort_index()

def mid_price(r,s,t,oc):
    b,a=float(r["best_bid"]),float(r["best_offer"])
    if np.isfinite(b) and np.isfinite(a) and b>0 and a>0: return 0.5*(b+a)
    if np.isfinite(b) and b>0: return b
    if np.isfinite(a) and a>0: return a
    vol=float(r["impl_volatility"]); vol=vol if np.isfinite(vol) and vol>0 else 0.20
    ty="call" if r["cp_flag"]=="C" else "put"
    return oc.black_scholes_price(s,r["strike_price"],t,RISK_FREE_RATE,vol,ty)

def pick_atm(df,flag,s):
    f=df[(df["cp_flag"]==flag)&df["strike_price"].between(0.95*s,1.05*s)]
    if f.empty: return None
    idx=(f["strike_price"]-s).abs().idxmin()
    r=f.loc[idx]
    return r if isinstance(r,pd.Series) else r.iloc[0]

def run(quiet=False,return_metrics=False,start_date=None,end_date=None,
        initial_cash=100_000_000.0):
    oc=OptionCalculator(); opts=load_options(DATA_CSV)
    
    # Load SPY underlying data from CSV
    raw=pd.read_csv("data/spy_underlying.csv", parse_dates=["datetime"])
    raw.set_index("datetime", inplace=True)
    close=raw["close"].astype(float).sort_index()
    
    if start_date is not None:
        close=close.loc[start_date:]; opts=opts.loc[start_date:]
    if end_date is not None:
        close=close.loc[:end_date];   opts=opts.loc[:end_date]
    idx=close.index.normalize().intersection(opts.index)
    close=close.loc[idx]; opts=opts.loc[idx]
    daily_opts={d:g for d,g in opts.groupby(opts.index.date)}
    cash=initial_cash; stock=SHARES_TO_HEDGE
    cash-=stock*close.iloc[0]
    call_ct=put_ct=0; call_meta=put_meta=None
    total_comm=total_slip=0.0
    rec,trade_pnls=[],[]
    for date in idx:
        s=close.loc[date]; day=daily_opts.get(date.date(),pd.DataFrame())
        if not day.empty:
            dte=(day["exdate"]-date).dt.days; day=day[dte.between(DTE_MIN,DTE_MAX)].copy()
        def close_leg(position,meta):
            nonlocal cash,total_comm,total_slip
            if position==0: return
            T=(meta[1]-date).days/TRADING_DAYS
            price=mid_price(meta[2],s,T,oc)
            pnl=(price-meta[3])*position*OPT_MULT
            trade_pnls.append(pnl)
            proceeds=price*position*OPT_MULT
            comm=abs(position)*COMMISSION_PER_CONTRACT
            slip=abs(proceeds)*SLIPPAGE_PERC
            total_comm+=comm; total_slip+=slip
            return proceeds-comm-slip
        if call_ct: cash+=close_leg(call_ct,call_meta); call_ct=0
        if put_ct:  cash+=close_leg(put_ct,put_meta);   put_ct=0
        if day.empty:
            rec.append({"date":date,"equity":cash+stock*s}); continue
        day["_T"]=(day["exdate"]-date).dt.days/TRADING_DAYS
        c_row,p_row=pick_atm(day,"C",s),pick_atm(day,"P",s)
        if c_row is None or p_row is None:
            rec.append({"date":date,"equity":cash+stock*s}); continue
        δc=oc.calculate_greeks(s,c_row["strike_price"],c_row["_T"],
                               RISK_FREE_RATE,float(c_row["impl_volatility"]),"call")["delta"]
        δp=oc.calculate_greeks(s,p_row["strike_price"],p_row["_T"],
                               RISK_FREE_RATE,float(p_row["impl_volatility"]),"put")["delta"]
        γc=oc.calculate_greeks(s,c_row["strike_price"],c_row["_T"],
                               RISK_FREE_RATE,float(c_row["impl_volatility"]),"call")["gamma"]
        γp=oc.calculate_greeks(s,p_row["strike_price"],p_row["_T"],
                               RISK_FREE_RATE,float(p_row["impl_volatility"]),"put")["gamma"]
        sol=np.linalg.lstsq(np.array([[δc,δp],[γc,γp]]),
                            np.array([-stock/OPT_MULT,0.0]),rcond=None)[0]
        call_ct=int(np.clip(np.round(sol[0]),-MAX_CONTRACTS,MAX_CONTRACTS))
        put_ct =int(np.clip(np.round(sol[1]),-MAX_CONTRACTS,MAX_CONTRACTS))
        entry_c=mid_price(c_row,s,c_row["_T"],oc)
        entry_p=mid_price(p_row,s,p_row["_T"],oc)
        cost_c=call_ct*entry_c*OPT_MULT; cost_p=put_ct*entry_p*OPT_MULT
        comm=(abs(call_ct)+abs(put_ct))*COMMISSION_PER_CONTRACT
        slip=(abs(cost_c)+abs(cost_p))*SLIPPAGE_PERC
        total_comm+=comm; total_slip+=slip
        cash-=cost_c+cost_p+comm+slip
        call_meta=(c_row["strike_price"],c_row["exdate"],c_row,entry_c)
        put_meta=(p_row["strike_price"],p_row["exdate"],p_row,entry_p)
        rec.append({"date":date,"equity":cash+stock*s+
                    call_ct*entry_c*OPT_MULT+put_ct*entry_p*OPT_MULT})
    df=pd.DataFrame(rec).set_index("date")
    ret=df["equity"].pct_change().dropna()
    ann_vol=ret.std()*sqrt(252)
    
    # Calculate comprehensive metrics focused on hedging effectiveness
    start_value = float(df["equity"].iloc[0])
    end_value = float(df["equity"].iloc[-1])
    total_return = (end_value - start_value) / start_value
    total_days = len(df)
    
    # Core hedging metrics - focus on volatility reduction and stability
    annual_volatility_pct = ann_vol * 100
    
    # Drawdown analysis - critical for hedging
    peak = df["equity"].expanding().max()
    drawdown = (df["equity"] - peak) / peak
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown * 100
    
    # Average drawdown duration and magnitude
    in_drawdown = drawdown < 0
    drawdown_periods = []
    current_dd_length = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dd_length += 1
        else:
            if current_dd_length > 0:
                drawdown_periods.append(current_dd_length)
            current_dd_length = 0
    if current_dd_length > 0:
        drawdown_periods.append(current_dd_length)
    
    avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0.0
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0.0
    
    # Stability metrics
    equity_std = df["equity"].std()
    equity_mean = df["equity"].mean()
    coefficient_of_variation = equity_std / equity_mean if equity_mean > 0 else float('inf')
    
    # Downside metrics focused on hedging
    negative_returns = ret[ret < 0]
    downside_deviation_pct = negative_returns.std() * sqrt(252) * 100 if len(negative_returns) > 0 else 0.0
    
    # Daily volatility in dollar terms
    daily_vol_dollar = ret.std() * start_value
    
    # Skewness and Kurtosis - important for tail risk
    skewness = ret.skew()
    kurtosis = ret.kurtosis()
    
    # VaR and CVaR (5% level) - tail risk measures
    var_5 = ret.quantile(0.05)
    cvar_5 = ret[ret <= var_5].mean()
    var_5_pct = var_5 * 100
    cvar_5_pct = cvar_5 * 100
    daily_var_5_dollar = var_5 * start_value
    daily_cvar_5_dollar = cvar_5 * start_value
    
    # Volatility consistency - how stable is the volatility over time
    rolling_vol = ret.rolling(window=21).std() * sqrt(252)  # 21-day rolling vol
    vol_of_vol = rolling_vol.std()  # Volatility of volatility
    
    # Equity curve flatness metrics
    equity_trend = np.polyfit(range(len(df["equity"])), df["equity"], 1)[0]  # Linear trend slope
    equity_range_pct = (df["equity"].max() - df["equity"].min()) / start_value * 100
    
    # Zero-crossing rate (how often does the equity curve cross its mean)
    equity_demeaned = df["equity"] - df["equity"].mean()
    zero_crossings = np.sum(np.diff(np.sign(equity_demeaned)) != 0)
    zero_crossing_rate = zero_crossings / len(df["equity"]) * 100
    
    # Hedging cost efficiency
    total_costs = total_comm + total_slip
    cost_drag_bps = (total_costs / start_value) * 10000  # basis points
    cost_per_vol_point = total_costs / (25.0 - ann_vol) if ann_vol < 25.0 else float('inf')  # Cost per % vol reduction
    vol_reduction_pct = max(0, (25.0 - ann_vol) / 25.0 * 100)  # Assuming 25% baseline vol
    
    # Trade analysis
    total_trade_pnl = sum(trade_pnls) if trade_pnls else 0.0
    num_trades = len(trade_pnls) if trade_pnls else 0
    avg_trade_pnl = total_trade_pnl / num_trades if num_trades > 0 else 0.0
    trade_frequency = num_trades / total_days * 252  # Annualized trade frequency
    
    # Risk-adjusted hedging effectiveness
    hedging_efficiency_ratio = vol_reduction_pct / cost_drag_bps if cost_drag_bps > 0 else 0.0  # Vol reduction per bp of cost
    
    # Ulcer Index - alternative to max drawdown that considers duration
    ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100
    
    # Pain Index - average drawdown
    pain_index = np.mean(np.abs(drawdown)) * 100
    
    metrics=dict(start_value=start_value,
                 end_value=end_value,
                 annual_volatility_pct=annual_volatility_pct,
                 max_drawdown_pct=max_drawdown_pct,
                 avg_drawdown_duration=avg_drawdown_duration,
                 max_drawdown_duration=max_drawdown_duration,
                 downside_deviation_pct=downside_deviation_pct,
                 coefficient_of_variation=coefficient_of_variation,
                 daily_vol_dollar=daily_vol_dollar,
                 skewness=skewness,
                 kurtosis=kurtosis,
                 var_5_pct=var_5_pct,
                 cvar_5_pct=cvar_5_pct,
                 daily_var_5_dollar=daily_var_5_dollar,
                 daily_cvar_5_dollar=daily_cvar_5_dollar,
                 vol_of_vol=vol_of_vol,
                 equity_trend=equity_trend,
                 equity_range_pct=equity_range_pct,
                 zero_crossing_rate=zero_crossing_rate,
                 total_commission_cost=total_comm,
                 total_slippage_cost=total_slip,
                 total_costs=total_costs,
                 cost_drag_bps=cost_drag_bps,
                 cost_per_vol_point=cost_per_vol_point,
                 vol_reduction_pct=vol_reduction_pct,
                 num_trades=num_trades,
                 avg_trade_pnl=avg_trade_pnl,
                 trade_frequency=trade_frequency,
                 total_trade_pnl=total_trade_pnl,
                 trading_days=total_days,
                 hedging_efficiency_ratio=hedging_efficiency_ratio,
                 ulcer_index=ulcer_index,
                 pain_index=pain_index)
    if not quiet:
        print(json.dumps(metrics,indent=2))
    if return_metrics: return metrics

if __name__=="__main__":
    run()
