# Script for defining rBergomi model parameters.

import numpy as np
import math
from utils import mean, variance, covariance, detrend_segment, hurst_exponent_DFA

# Calculates log returns from price data.
def log_returns(prices):
    prices = np.array(prices, dtype=np.float64)
    if prices.size < 2:
        return np.array([])
    return np.log(prices[1:] / prices[:-1])

# Estimates xi parameter from log returns.
def estimate_xi(logrets, dt_yr):
    var_r = variance(logrets)
    return var_r / dt_yr

# Estimates Hurst exponent from log returns.
def estimate_H(logrets):
    return hurst_exponent_DFA(logrets)

# Estimates eta parameter from log returns.
def estimate_eta(logrets, H, window=20):
    if len(logrets) < window:
        stdev = math.sqrt(np.var(logrets, ddof=1))
        return stdev * 2.0
    realized_var = []
    for i in range(window - 1, len(logrets)):
        window_returns = logrets[i - window + 1:i + 1]
        rv = np.mean(np.square(window_returns))
        realized_var.append(rv)
    log_rv = np.log(np.array(realized_var))
    log_diff = np.diff(log_rv)
    daily_eta = np.std(log_diff, ddof=1)
    return daily_eta * math.sqrt(252)

# Estimates rho parameter from log returns.
def estimate_rho(logrets):
    logrets = np.array(logrets, dtype=np.float64)
    sq = logrets ** 2
    c = covariance(logrets, sq)
    denom = math.sqrt(variance(logrets) * variance(sq))
    rho = c / denom if denom != 0.0 else 0.0
    if rho > 0.0:
        rho = -0.3
    return rho

def estimate_params(historical_prices, dt_yr=1/252):
    prices = np.array(historical_prices, dtype=np.float64)
    rets = log_returns(prices)
    xi = estimate_xi(rets, dt_yr)
    H = estimate_H(rets)
    eta = estimate_eta(rets, H)
    rho = estimate_rho(rets)
    return xi, H, eta, rho
