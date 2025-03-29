# Utility functions for simulations and data processing.

import random
import numpy as np
import torch

# Calculates the mean of a vector.
def mean(v):
    return np.mean(v) if len(v) > 0 else 0.0

# Calculates the variance of a vector.
def variance(v):
    if len(v) < 2:
        return 0.0
    return np.var(v, ddof=1)

# Calculates the covariance between two vectors.
def covariance(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return np.cov(x, y, ddof=1)[0, 1]

# Detrends a segment of data.
def detrend_segment(segment):
    n = len(segment)
    if n < 2:
        return segment
    t = np.arange(1, n+1, dtype=np.float64)
    tm = mean(t)
    ym = mean(segment)
    num = np.sum((t - tm) * (segment - ym))
    den = np.sum((t - tm) ** 2)
    if abs(den) < 1e-14:
        return segment
    slope = num / den
    intercept = ym - slope * tm
    return segment - (slope * t + intercept)

# Calculates the Hurst exponent using DFA.
def hurst_exponent_DFA(data_in):
    data = np.array(data_in, dtype=np.float64)
    if len(data) < 2:
        return 0.5
    data = data - mean(data)
    data = np.cumsum(data)
    log_window_size = []
    log_fluctuation = []
    min_window_size = 4
    max_window_size = len(data) // 4
    w = min_window_size
    while w <= max_window_size:
        fluctuations = []
        for start in range(0, len(data) - w + 1, w):
            segment = data[start:start+w]
            detrended = detrend_segment(segment)
            rms = np.sqrt(np.mean(detrended**2))
            fluctuations.append(rms)
        mf = mean(fluctuations)
        if mf > 0.0:
            log_window_size.append(np.log(w))
            log_fluctuation.append(np.log(mf))
        w *= 2
    n = len(log_window_size)
    if n < 2:
        return 0.5
    sumX = np.sum(log_window_size)
    sumY = np.sum(log_fluctuation)
    sumXX = np.sum(np.array(log_window_size) ** 2)
    sumXY = np.sum(np.array(log_window_size) * np.array(log_fluctuation))
    slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX ** 2)
    return slope

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
