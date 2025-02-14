# Script for assigning option prices in simulations.

import numpy as np
import math
from scipy.stats import norm

INPUT_PATHS_FILE = './data/paths.npy'
OUTPUT_OPTIONS_FILE = './data/paths_options.npz'
RISK_FREE_RATE = 0.04

# Calculates Black-Scholes prices for options using vectorized operations.
def black_scholes_vectorized(S, K, T, r, sigma, epsilon=1e-8):
    T_safe = np.where(T <= 0, 1e-8, T)
    sigma_safe = np.where(sigma < epsilon, epsilon, sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    call = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    put = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)
    intrinsic_call = np.maximum(S - K * np.exp(-r * T), 0)
    intrinsic_put = np.maximum(K * np.exp(-r * T) - S, 0)
    call = np.where(T <= 0, intrinsic_call, call)
    put = np.where(T <= 0, intrinsic_put, put)
    return call, put

# Calculates annualized volatility matrix for paths.
def calculate_annualized_vol_matrix(paths):
    n_sims, n_steps1 = paths.shape
    vols = np.zeros((n_sims, n_steps1))
    for t in range(1, n_steps1):
        slice_paths = paths[:, :t+1]
        log_rets = np.log(slice_paths[:, 1:] / slice_paths[:, :-1])
        sigma_daily = np.std(log_rets, axis=1, ddof=1)
        vols[:, t] = sigma_daily * math.sqrt(252)
    return vols

def process_price_paths():
    paths = np.load(INPUT_PATHS_FILE)
    n_sims, n_steps1 = paths.shape
    strikes = np.round(paths[:, 0])
    time_grid = np.arange(n_steps1)
    T = np.clip(1 - time_grid / 252, 0, None)

    vols = calculate_annualized_vol_matrix(paths)
    calls = np.zeros((n_sims, n_steps1))
    puts = np.zeros((n_sims, n_steps1))

    for t in range(n_steps1):
        S = paths[:, t]
        sigma = vols[:, t]
        call_t, put_t = black_scholes_vectorized(S, strikes, T[t], RISK_FREE_RATE, sigma)
        calls[:, t] = call_t
        puts[:, t] = put_t

    np.savez(OUTPUT_OPTIONS_FILE, calls=calls, puts=puts)
    print(f"Processed {n_sims} simulations and saved option prices to {OUTPUT_OPTIONS_FILE}")

if __name__ == '__main__':
    process_price_paths()
