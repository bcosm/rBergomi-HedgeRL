import backtrader as bt
import pandas as pd
import numpy as np
from datetime import timedelta
from math import sqrt
from model_wrapper_bt import ModelWrapper
from option_calculator import OptionCalculator
from tqdm import tqdm
import json, warnings, random, torch
from types import SimpleNamespace
from delta_hedge import run as run_delta_hedge

warnings.filterwarnings("ignore",
                        message=r"The behavior of DataFrame concatenation.*",
                        category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
#  RUN-CONTROL FLAGS
# ──────────────────────────────────────────────────────────────────────────────
USE_LOOP         = True      # False → single run with SINGLE_SEED
NUM_SEEDS        = 1         # loop count if USE_LOOP = True
SINGLE_SEED      = 12        # seed when USE_LOOP = False
# ──────────────────────────────────────────────────────────────────────────────

class OptionCommission(bt.CommissionInfo):
    params = dict(commission=0.0, mult=100.0)
    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission

class SPYReturn(bt.Observer):
    lines = ('spyret',)
    plotinfo = dict(plot=True, subplot=True, plotname='SPY Return %')
    def start(self):
        strat = self._owner
        self.init_spy = strat.spy.close[0] * strat.p.shares_to_hedge
    def next(self):
        strat = self._owner
        today = strat.spy.datetime.date(0)
        cash = strat.broker.get_cash()
        spy_val = strat.getposition(strat.spy).size * strat.spy.close[0]
        opt_val = strat.current_option_value(today)
        total_equity = cash + spy_val + opt_val
        self.lines.spyret[0] = (total_equity - self.init_spy) / self.init_spy * 100.0

class ManualValue(bt.Observer):
    lines = ('equity',)
    plotinfo = dict(plot=True, subplot=True, plotname='Equity')
    def next(self):
        strat = self._owner
        today = strat.spy.datetime.date(0)
        cash = strat.broker.get_cash()
        spy_val = strat.getposition(strat.spy).size * strat.spy.close[0]
        opt_val = strat.current_option_value(today)
        total_equity = cash + spy_val + opt_val
        self.lines.equity[0] = total_equity
        strat.equity_history.append((pd.to_datetime(today), total_equity))

class DRLHedgingStrategy(bt.Strategy):
    params = dict(shares_to_hedge=10000,max_contracts_per_type=200,
                  max_trade_per_step=100,option_tenor_years=30/252,
                  risk_free_rate=0.02,episode_length=252,warmup_period=5,
                  artifacts_path='model_files/',options_df=None,
                  commission_per_contract=0.65,slippage_perc=0.001,
                  expiry_roll_days=7,prog_ns=None)
    def __init__(self):
        self.spy=self.datas[0]
        self.options_df=self.p.options_df.copy()
        self.options_df['strike_price']/=1000.0
        self.options_df['best_bid']=self.options_df['best_bid'].astype(float)
        self.options_df['best_offer']=self.options_df['best_offer'].astype(float)
        self.options_df['exdate']=pd.to_datetime(self.options_df['exdate'])
        self.options_by_date={d:g for d,g in self.options_df.groupby(self.options_df.index.date)}
        self.model_wrapper=ModelWrapper(artifacts_path=self.p.artifacts_path,logger=self.log)
        self.opt_calc=OptionCalculator()
        self.option_positions=pd.DataFrame(columns=['type','strike','expiry','qty','entry_price'])
        self.trade_pnls=[]
        self.equity_history=[]
        self.total_commission_cost=0.0
        self.total_slippage_cost=0.0
        self.current_step=0
        self.initial_S0_for_episode=None
        self.S_t_minus_1=None
        self.v_t_minus_1=None
        self.last_price=None
        self.last_vol=None
    def get_observation(self,df):
        if self.initial_S0_for_episode is None:
            self.initial_S0_for_episode=self.last_price
        S,v=self.last_price,self.last_vol
        C,P=self.get_atm_option_prices(S,df)
        if C is None or P is None:
            return None
        s0=max(self.initial_S0_for_episode,25.0)
        calls,puts=self.get_current_contract_counts()
        delta_t=max(0.0,(self.p.episode_length-self.current_step)/self.p.episode_length)
        K=round(S); sigma=np.sqrt(v)
        cg=self.opt_calc.calculate_greeks(S,K,self.p.option_tenor_years,
                                          self.p.risk_free_rate,sigma,'call')
        pg=self.opt_calc.calculate_greeks(S,K,self.p.option_tenor_years,
                                          self.p.risk_free_rate,sigma,'put')
        call_delta,call_gamma=cg['delta'],cg['gamma']
        put_delta,put_gamma=pg['delta'],cg['gamma']
        lag_S=np.clip((S-self.S_t_minus_1)/self.S_t_minus_1,-1.,1.) if self.S_t_minus_1 else 0.0
        lag_v=np.clip(v-self.v_t_minus_1,-1.,1.) if self.v_t_minus_1 is not None else 0.0
        return np.array([S/s0,C/s0,P/s0,calls/self.p.max_contracts_per_type,
                         puts/self.p.max_contracts_per_type,v,delta_t,call_delta,
                         call_gamma,put_delta,put_gamma,lag_S,lag_v],dtype=np.float32)
    def log(self,txt,dt=None):
        if txt.startswith("Actor weights loaded"): return
        dt=dt or self.spy.datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')
    def prenext(self): self.last_price=float(self.spy.close[0])
    def next(self):
        if self.p.prog_ns and self.p.prog_ns.bar: self.p.prog_ns.bar.update(1)
        if len(self)<=self.p.warmup_period:
            self.S_t_minus_1=float(self.spy.close[0]); return
        if not getattr(self,'spy_hedged',False):
            self.order_target_size(self.spy,target=self.p.shares_to_hedge)
            self.spy_hedged=True
        self.roll_near_expiry_options(); self.daily_rebalance()
    def update_implied_volatility(self,df):
        ivs=df[df['impl_volatility']>0]
        atm=ivs[np.abs(ivs['strike_price']-self.last_price)<self.last_price*0.05]['impl_volatility']
        if len(atm)>=3: self.last_vol=float(np.mean(np.square(atm)))
    def validate_market_data(self):
        if self.last_price is None: return False
        if self.last_vol is None: self.last_vol=0.20**2
        return True
    def get_current_contract_counts(self):
        c=self.option_positions[self.option_positions['type']=='C']['qty'].sum()
        p=self.option_positions[self.option_positions['type']=='P']['qty'].sum()
        return int(c),int(p)
    def _nearest_option(self,df,strike):
        return None if df.empty else df.iloc[(df['strike_price']-strike).abs().values.argmin()]
    def _update_option_position(self,typ,data,qty):
        price=0.5*(data['best_bid']+data['best_offer'])
        cost=price*qty*100
        comm=abs(qty)*self.p.commission_per_contract
        slip=abs(cost)*self.p.slippage_perc
        self.total_commission_cost+=comm
        self.total_slippage_cost+=slip
        self.broker.set_cash(self.broker.get_cash()-cost-comm-slip)
        mask=(self.option_positions['type']==typ)&(self.option_positions['strike']==data['strike_price'])&(self.option_positions['expiry']==data['exdate'])
        if not self.option_positions[mask].empty:
            idx=self.option_positions[mask].index[0]
            prev_qty=self.option_positions.at[idx,'qty']
            new_qty=prev_qty+qty
            if new_qty==0:
                entry=self.option_positions.loc[idx]
                pnl=(price-entry['entry_price'])*prev_qty*100
                self.trade_pnls.append(pnl)
                self.option_positions.drop(idx,inplace=True)
            else:
                self.option_positions.at[idx,'qty']=new_qty
        else:
            row=dict(type=typ,strike=data['strike_price'],expiry=data['exdate'],
                     qty=qty,entry_price=price)
            self.option_positions=pd.concat([self.option_positions,
                                             pd.DataFrame([row])],ignore_index=True)
    def _liquidate_option(self,idx,data):
        pos=self.option_positions.loc[idx]
        price=0.5*(data['best_bid']+data['best_offer'])
        proceeds=price*pos['qty']*100
        comm=abs(pos['qty'])*self.p.commission_per_contract
        slip=abs(proceeds)*self.p.slippage_perc
        self.total_commission_cost+=comm
        self.total_slippage_cost+=slip
        self.broker.set_cash(self.broker.get_cash()+proceeds-comm-slip)
        pnl=(price-pos['entry_price'])*pos['qty']*100
        self.trade_pnls.append(pnl)
        self.option_positions.drop(idx,inplace=True)
    def current_option_value(self,today):
        if self.option_positions.empty: return 0.0
        df=self.options_by_date.get(today,pd.DataFrame()); total=0.0
        for _,p in self.option_positions.iterrows():
            subset=df[(df['strike_price']==p['strike'])&(df['exdate'].dt.date==p['expiry'].date())&(df['cp_flag']==p['type'])]
            if subset.empty: continue
            row=subset.iloc[0]; b,a=row['best_bid'],row['best_offer']
            mid=0.5*(b+a) if (b>0 and a>0) else (b if b>0 else a)
            if mid<=0:
                T=(p['expiry'].date()-today).days/365.0
                sigma=np.sqrt(self.last_vol or 0.20**2)
                mid=self.opt_calc.black_scholes_price(self.spy.close[0],p['strike'],T,
                                                      self.p.risk_free_rate,sigma,
                                                      option_type='call' if p['type']=='C' else 'put')
            total+=mid*p['qty']*100
        return total
    def get_atm_option_prices(self,S,df):
        def mid_px(r):
            b,a=r['best_bid'],r['best_offer']
            if b>0 and a>0: return 0.5*(b+a)
            if b>0: return b
            if a>0: return a
            T=(r['exdate'].date()-pd.to_datetime(self.spy.datetime.date(0)).date()).days/365.0
            sigma=np.sqrt(self.last_vol or 0.20**2)
            return self.opt_calc.black_scholes_price(S,r['strike_price'],T,
                                                     self.p.risk_free_rate,sigma,
                                                     option_type='call' if r['cp_flag']=='C' else 'put')
        df2=df.copy(); df2['mid_price']=df2.apply(mid_px,axis=1)
        df2=df2.dropna(subset=['mid_price'])
        call_row=self._nearest_option(df2[df2['cp_flag']=='C'],round(S))
        put_row=self._nearest_option(df2[df2['cp_flag']=='P'],round(S))
        return (None,None) if (call_row is None or put_row is None) else \
               (float(call_row['mid_price']),float(put_row['mid_price']))
    def daily_rebalance(self):
        today=pd.to_datetime(self.spy.datetime.date(0))
        if getattr(self,'last_rebalance_date',None) and today<=self.last_rebalance_date: return
        self.last_rebalance_date=today
        self.last_price=float(self.spy.close[0])
        df=self.options_by_date.get(today.date(),pd.DataFrame())
        if df.empty: self.advance_step(); return
        dte=(df['exdate']-today).dt.days
        df=df[(dte>=10)&(dte<=45)]
        if df.empty: self.advance_step(); return
        strikes=sorted(df['strike_price'].unique())
        idx=min(range(len(strikes)),key=lambda i:abs(strikes[i]-self.last_price))
        df=df[df['strike_price'].isin(set(strikes[max(0,idx-10):idx+11]))]
        df=df[np.abs(df['strike_price']-self.last_price)<=10]
        if df.empty: self.advance_step(); return
        self.update_implied_volatility(df)
        if not self.validate_market_data(): self.advance_step(); return
        obs=self.get_observation(df)
        if obs is None: self.advance_step(); return
        self.model_wrapper.model.eval()
        with torch.no_grad():
            actions=self.model_wrapper.predict(obs)
        if actions is None: actions=np.array([0.0,0.0],dtype=np.float32)
        self.execute_option_trades(actions,df); self.advance_step()
    def execute_option_trades(self,actions,df):
        calls_now,puts_now=self.get_current_contract_counts()
        ct=int(np.rint(np.clip(actions[0],-1,1)*self.p.max_trade_per_step))
        pt=int(np.rint(np.clip(actions[1],-1,1)*self.p.max_trade_per_step))
        new_c=np.clip(calls_now+ct,-self.p.max_contracts_per_type,self.p.max_contracts_per_type)
        new_p=np.clip(puts_now+pt,-self.p.max_contracts_per_type,self.p.max_contracts_per_type)
        act_c,act_p=int(new_c-calls_now),int(new_p-puts_now)
        if act_c:
            opt=self._nearest_option(df[df['cp_flag']=='C'],round(self.last_price))
            if opt is not None: self._update_option_position('C',opt,act_c)
        if act_p:
            opt=self._nearest_option(df[df['cp_flag']=='P'],round(self.last_price))
            if opt is not None: self._update_option_position('P',opt,act_p)
    def roll_near_expiry_options(self):
        if self.option_positions.empty: return
        today=pd.to_datetime(self.spy.datetime.date(0))
        df=self.options_by_date.get(today.date(),pd.DataFrame())
        for idx,pos in self.option_positions.copy().iterrows():
            if (pos['expiry'].date()-today.date()).days<self.p.expiry_roll_days:
                subset=df[(df['strike_price']==pos['strike'])&
                          (df['exdate'].dt.date==pos['expiry'].date())&
                          (df['cp_flag']==pos['type'])]
                if subset.empty: continue
                self._liquidate_option(idx,subset.iloc[0])
    def advance_step(self):
        self.S_t_minus_1,self.v_t_minus_1=self.last_price,self.last_vol
        self.current_step+=1
        if self.current_step>=self.p.episode_length: self.reset_episode()
    def reset_episode(self):
        today=pd.to_datetime(self.spy.datetime.date(0))
        df=self.options_by_date.get(today.date(),pd.DataFrame())
        for idx,pos in self.option_positions.copy().iterrows():
            subset=df[(df['strike_price']==pos['strike'])&(df['exdate'].dt.date==pos['expiry'].date())&(df['cp_flag']==pos['type'])]
            if not subset.empty: self._liquidate_option(idx,subset.iloc[0])
            else: self.option_positions.drop(idx,inplace=True)
        self.option_positions=self.option_positions.iloc[0:0]
        self.order_target_size(self.spy,target=self.p.shares_to_hedge)
        self.model_wrapper.reset_hidden_states()
        self.current_step=0
        self.initial_S0_for_episode=float(self.spy.close[0])
        self.S_t_minus_1=None; self.v_t_minus_1=None

def one_run(seed):
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    cerebro=bt.Cerebro(stdstats=False)
    cerebro.addobserver(ManualValue,subplot=True)
    cerebro.addobserver(SPYReturn,subplot=True)
    opts=pd.read_csv('data/spy_options.csv',
                     usecols=['date','exdate','cp_flag','strike_price',
                              'best_bid','best_offer','impl_volatility'],
                     parse_dates=['date','exdate'])
    opts.set_index('date',inplace=True); opts.sort_index(inplace=True)
    start,end=opts.index.min(),opts.index.max()
    
    # Load SPY underlying data from CSV
    spy=pd.read_csv('data/spy_underlying.csv', parse_dates=['datetime'])
    spy.set_index('datetime',inplace=True)
    spy.index.name='datetime'
    spy_bt=spy.loc['2008-01-01':'2023-01-01'].copy()
    prog_ns=SimpleNamespace(bar=tqdm(total=len(spy_bt),desc=f"Seed {seed}"))
    cerebro.adddata(bt.feeds.PandasData(dataname=spy_bt))
    cerebro.addstrategy(DRLHedgingStrategy,options_df=opts,prog_ns=prog_ns)
    cerebro.broker.setcash(100_000_000.0)
    cerebro.broker.setcommission(commission=0.65/100,mult=100.0)
    strat=cerebro.run()[0]
    prog_ns.bar.close()
    equity_series=pd.Series(dict(strat.equity_history)).sort_index()
    returns_series=equity_series.pct_change().dropna()
    ann_vol=returns_series.std()*sqrt(252)
    
    # Calculate comprehensive metrics focused on hedging effectiveness
    start_value = float(equity_series.iloc[0])
    end_value = float(equity_series.iloc[-1])
    total_return = (end_value - start_value) / start_value
    total_days = len(equity_series)
    
    # Core hedging metrics - focus on volatility reduction and stability
    annual_volatility_pct = ann_vol * 100
    
    # Drawdown analysis - critical for hedging
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
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
    equity_std = equity_series.std()
    equity_mean = equity_series.mean()
    coefficient_of_variation = equity_std / equity_mean if equity_mean > 0 else float('inf')
    
    # Downside metrics focused on hedging
    negative_returns = returns_series[returns_series < 0]
    downside_deviation_pct = negative_returns.std() * sqrt(252) * 100 if len(negative_returns) > 0 else 0.0
    
    # Daily volatility in dollar terms
    daily_vol_dollar = returns_series.std() * start_value
    
    # Skewness and Kurtosis - important for tail risk
    skewness = returns_series.skew()
    kurtosis = returns_series.kurtosis()
    
    # VaR and CVaR (5% level) - tail risk measures
    var_5 = returns_series.quantile(0.05)
    cvar_5 = returns_series[returns_series <= var_5].mean()
    var_5_pct = var_5 * 100
    cvar_5_pct = cvar_5 * 100
    daily_var_5_dollar = var_5 * start_value
    daily_cvar_5_dollar = cvar_5 * start_value
    
    # Volatility consistency - how stable is the volatility over time
    rolling_vol = returns_series.rolling(window=21).std() * sqrt(252)  # 21-day rolling vol
    vol_of_vol = rolling_vol.std()  # Volatility of volatility
    
    # Equity curve flatness metrics
    equity_trend = np.polyfit(range(len(equity_series)), equity_series, 1)[0]  # Linear trend slope
    equity_range_pct = (equity_series.max() - equity_series.min()) / start_value * 100
    
    # Zero-crossing rate (how often does the equity curve cross its mean)
    equity_demeaned = equity_series - equity_series.mean()
    zero_crossings = np.sum(np.diff(np.sign(equity_demeaned)) != 0)
    zero_crossing_rate = zero_crossings / len(equity_series) * 100
    
    # Hedging cost efficiency
    total_costs = strat.total_commission_cost + strat.total_slippage_cost
    cost_drag_bps = (total_costs / start_value) * 10000  # basis points
    cost_per_vol_point = total_costs / (25.0 - ann_vol) if ann_vol < 25.0 else float('inf')  # Cost per % vol reduction
    vol_reduction_pct = max(0, (25.0 - ann_vol) / 25.0 * 100)  # Assuming 25% baseline vol
    
    # Trade analysis
    total_trade_pnl = sum(strat.trade_pnls) if strat.trade_pnls else 0.0
    num_trades = len(strat.trade_pnls) if strat.trade_pnls else 0
    avg_trade_pnl = total_trade_pnl / num_trades if num_trades > 0 else 0.0
    trade_frequency = num_trades / total_days * 252  # Annualized trade frequency
    
    # Risk-adjusted hedging effectiveness
    hedging_efficiency_ratio = vol_reduction_pct / cost_drag_bps if cost_drag_bps > 0 else 0.0  # Vol reduction per bp of cost
    
    # Ulcer Index - alternative to max drawdown that considers duration
    ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100
    
    # Pain Index - average drawdown
    pain_index = np.mean(np.abs(drawdown)) * 100
    
    metrics=dict(seed=seed,
                 start_value=start_value,
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
                 total_commission_cost=strat.total_commission_cost,
                 total_slippage_cost=strat.total_slippage_cost,
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
    
    # Run delta hedge comparison for single run mode
    delta=run_delta_hedge(quiet=True,return_metrics=True,
                          start_date=pd.to_datetime('2008-01-01').date(),
                          end_date=pd.to_datetime('2023-01-01').date(),
                          initial_cash=100_000_000.0)
    
    print("\n" + "="*80)
    print("=== COMPREHENSIVE HEDGING PERFORMANCE COMPARISON ===")
    print("="*80)
    
    print("\n=== RL Agent Metrics ===")
    print(json.dumps(metrics,indent=2))
    
    print("\n=== Delta-Hedge Baseline ===")
    print(json.dumps(delta,indent=2))
    
    print("\n" + "="*80)
    print("=== KEY PERFORMANCE COMPARISON ===")
    print("="*80)
    
    # Create comparison table for key hedging metrics
    comparison_metrics = [
        'annual_volatility_pct', 'max_drawdown_pct', 'downside_deviation_pct', 
        'ulcer_index', 'pain_index', 'total_costs', 'cost_drag_bps',
        'vol_reduction_pct', 'cost_per_vol_point', 'hedging_efficiency_ratio',
        'vol_of_vol', 'equity_range_pct', 'zero_crossing_rate',
        'avg_drawdown_duration', 'max_drawdown_duration', 'trade_frequency'
    ]
    
    print(f"{'Metric':<25} {'RL Agent':<15} {'Delta Hedge':<15} {'RL Advantage':<15}")
    print("-" * 70)
    
    for metric in comparison_metrics:
        if metric in metrics and metric in delta:
            rl_val = metrics[metric]
            delta_val = delta[metric]
            
            # Calculate advantage (positive means RL is better)
            if metric in ['annual_volatility_pct', 'max_drawdown_pct', 'downside_deviation_pct', 
                         'ulcer_index', 'pain_index', 'total_costs', 'cost_drag_bps',
                         'vol_of_vol', 'equity_range_pct', 'avg_drawdown_duration', 
                         'max_drawdown_duration', 'cost_per_vol_point', 'trade_frequency']:
                # Lower is better for these metrics
                advantage = ((delta_val - rl_val) / abs(delta_val)) * 100 if delta_val != 0 else 0
            else:
                # Higher is better for these metrics (vol_reduction_pct, hedging_efficiency_ratio, zero_crossing_rate)
                advantage = ((rl_val - delta_val) / abs(delta_val)) * 100 if delta_val != 0 else 0
            
            print(f"{metric:<25} {rl_val:<15.2f} {delta_val:<15.2f} {advantage:<15.1f}%")
    
    print("\n" + "="*80)
    
    return metrics

def loop_runs():
    all_metrics=[]
    for s in range(NUM_SEEDS):
        m = one_run(109+s)
        all_metrics.append(m)
    df=pd.DataFrame(all_metrics)
    avg=df.mean(numeric_only=True)
    best=df.iloc[df['annual_volatility_pct'].idxmin()]
    delta=run_delta_hedge(quiet=True,return_metrics=True,
                          start_date=pd.to_datetime('2008-01-01').date(),
                          end_date=pd.to_datetime('2023-01-01').date(),
                          initial_cash=100_000_000.0)
    
    print("\n" + "="*80)
    print("=== COMPREHENSIVE HEDGING PERFORMANCE COMPARISON ===")
    print("="*80)
    
    print("\n=== Average RL Metrics Across Seeds ===")
    print(json.dumps(avg.to_dict(),indent=2))
    
    print("\n=== Best-Volatility RL Seed ===")
    print(json.dumps(best.to_dict(),indent=2))
    
    print("\n=== Delta-Hedge Baseline ===")
    print(json.dumps(delta,indent=2))
    
    print("\n" + "="*80)
    print("=== KEY PERFORMANCE COMPARISON ===")
    print("="*80)
    
    # Create comparison table for key hedging metrics
    comparison_metrics = [
        'annual_volatility_pct', 'max_drawdown_pct', 'downside_deviation_pct', 
        'ulcer_index', 'pain_index', 'total_costs', 'cost_drag_bps',
        'vol_reduction_pct', 'cost_per_vol_point', 'hedging_efficiency_ratio',
        'vol_of_vol', 'equity_range_pct', 'zero_crossing_rate',
        'avg_drawdown_duration', 'max_drawdown_duration', 'trade_frequency'
    ]
    
    print(f"{'Metric':<25} {'Avg RL':<15} {'Best RL':<15} {'Delta Hedge':<15} {'RL Advantage':<15}")
    print("-" * 85)
    
    for metric in comparison_metrics:
        if metric in avg and metric in best and metric in delta:
            avg_val = avg[metric]
            best_val = best[metric]
            delta_val = delta[metric]
            
            # Calculate advantage (positive means RL is better)
            if metric in ['annual_volatility_pct', 'max_drawdown_pct', 'downside_deviation_pct', 
                         'ulcer_index', 'pain_index', 'total_costs', 'cost_drag_bps',
                         'vol_of_vol', 'equity_range_pct', 'avg_drawdown_duration', 
                         'max_drawdown_duration', 'cost_per_vol_point', 'trade_frequency']:
                # Lower is better for these metrics
                advantage = ((delta_val - best_val) / abs(delta_val)) * 100 if delta_val != 0 else 0
            else:
                # Higher is better for these metrics (vol_reduction_pct, hedging_efficiency_ratio, zero_crossing_rate)
                advantage = ((best_val - delta_val) / abs(delta_val)) * 100 if delta_val != 0 else 0
            
            print(f"{metric:<25} {avg_val:<15.2f} {best_val:<15.2f} {delta_val:<15.2f} {advantage:<15.1f}%")
    
    print("\n" + "="*80)

if __name__=='__main__':
    if USE_LOOP:
        loop_runs()
    else:
        one_run(SINGLE_SEED)
