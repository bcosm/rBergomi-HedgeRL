# Simulation script for generating paths and options using the rBergomi model.
# This script estimates the rBergomi model parameters from historical prices,
# generates synthetic paths for the underlying asset, and prices European-style
# call and put options using Monte Carlo simulation.

import time
import random
import os
import cupy as cp
import numpy as np
from tqdm import tqdm

INPUT_FILE = './data/historical_prices.csv'
OUTPUT_FILE = './data/paths_rbergomi_options_100k.npz'
CHECKPOINT_FILE = './rbergomi_generator_checkpoint.npz'
CHECKPOINT_FILE_TEMP_STEM = './rbergomi_generator_checkpoint_temp'

R = 0.04
DT = 1/252
N_PATHS = 100000
N_STEPS = 252
SEED = 42

T_OPTION_TENOR = 30/252
N_PATHS_OPTION_MC = 5000
OPTION_PRICING_MINI_BATCH_SIZE = 512

XI_DEFAULT = 0.04
H_DEFAULT = 0.1
ETA_DEFAULT = 1.0
RHO_DEFAULT = -0.7
S0_DEFAULT = 100.0

PERTURB_S0_STD = 0.01
PERTURB_XI_STD = 0.20
PERTURB_H_STD = 0.20
PERTURB_ETA_STD = 0.20
PERTURB_RHO_STD = 0.10

MIN_XI_FACTOR = 0.5
MIN_ETA_FACTOR = 0.5
CLIP_H_MIN = 0.01
CLIP_H_MAX = 0.49
CLIP_RHO_MIN = -0.99
CLIP_RHO_MAX = -0.01


def np_mean(v):
    return np.mean(v) if len(v) > 0 else 0.0

def np_variance(v):
    if len(v) < 2:
        return 0.0
    return np.var(v, ddof=1)

def np_covariance(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return np.cov(x, y, ddof=1)[0, 1]

def log_returns(prices_np):
    prices_np = np.array(prices_np, dtype=np.float64)
    if prices_np.size < 2:
        return np.array([])
    return np.log(prices_np[1:] / prices_np[:-1])

def estimate_xi(logrets_np, dt_yr_np):
    var_r_np = np_variance(logrets_np)
    return var_r_np / dt_yr_np

def detrend_segment(segment_np):
    n_np = len(segment_np)
    if n_np < 2:
        return segment_np
    t_np = np.arange(1, n_np + 1, dtype=np.float64)
    tm_np = np_mean(t_np)
    ym_np = np_mean(segment_np)
    num_np = np.sum((t_np - tm_np) * (segment_np - ym_np))
    den_np = np.sum((t_np - tm_np) ** 2)
    if abs(den_np) < 1e-14:
        return segment_np
    slope_np = num_np / den_np
    intercept_np = ym_np - slope_np * tm_np
    return segment_np - (slope_np * t_np + intercept_np)

def hurst_exponent_DFA(data_in_np):
    data_np = np.array(data_in_np, dtype=np.float64)
    if len(data_np) < 20:
        return H_DEFAULT
    data_np = data_np - np_mean(data_np)
    data_np = np.cumsum(data_np)
    log_window_size_np = []
    log_fluctuation_np = []
    min_window_size_np = 10
    max_window_size_np = len(data_np) // 4
    if max_window_size_np < min_window_size_np:
        return H_DEFAULT
        
    w_np = min_window_size_np
    while w_np <= max_window_size_np:
        fluctuations_np = []
        for start_np in range(0, len(data_np) - w_np + 1, w_np):
            segment_np = data_np[start_np:start_np+w_np]
            detrended_np = detrend_segment(segment_np)
            rms_np = np.sqrt(np.mean(detrended_np**2))
            if rms_np > 1e-8 : fluctuations_np.append(rms_np)
        if not fluctuations_np:
            if w_np == max_window_size_np: break
            if w_np * 2 > max_window_size_np and w_np < max_window_size_np : w_np = max_window_size_np
            elif w_np * 2 > max_window_size_np and w_np == max_window_size_np: break
            else: w_np *= 2
            continue

        mf_np = np_mean(fluctuations_np)
        if mf_np > 1e-8:
            log_window_size_np.append(np.log(w_np))
            log_fluctuation_np.append(np.log(mf_np))
        
        if w_np == max_window_size_np: break
        if w_np * 2 > max_window_size_np and w_np < max_window_size_np : w_np = max_window_size_np
        elif w_np * 2 > max_window_size_np and w_np == max_window_size_np: break
        else: w_np *= 2
            
    n_np_hurst = len(log_window_size_np)
    if n_np_hurst < 2:
        return H_DEFAULT
    sumX_np = np.sum(log_window_size_np)
    sumY_np = np.sum(log_fluctuation_np)
    sumXX_np = np.sum(np.array(log_window_size_np) ** 2)
    sumXY_np = np.sum(np.array(log_window_size_np) * np.array(log_fluctuation_np))
    denominator = (n_np_hurst * sumXX_np - sumX_np ** 2)
    if abs(denominator) < 1e-14:
        return H_DEFAULT
    slope_np = (n_np_hurst * sumXY_np - sumX_np * sumY_np) / denominator
    return np.clip(slope_np, CLIP_H_MIN, CLIP_H_MAX)

def estimate_H(logrets_np):
    return hurst_exponent_DFA(logrets_np)

def estimate_eta(logrets_np, H_np, window_np=20):
    if len(logrets_np) < window_np +1 :
        return ETA_DEFAULT
    realized_var_np = []
    for i_np in range(window_np - 1, len(logrets_np)):
        window_returns_np = logrets_np[i_np - window_np + 1:i_np + 1]
        rv_np = np.mean(np.square(window_returns_np))
        realized_var_np.append(rv_np)
    if not realized_var_np:
        return ETA_DEFAULT
    log_rv_np = np.log(np.array(realized_var_np))
    if len(log_rv_np) < 2:
        return ETA_DEFAULT
    log_diff_np = np.diff(log_rv_np)
    if len(log_diff_np) < 2:
        return ETA_DEFAULT
    daily_eta_np = np.std(log_diff_np, ddof=1)
    return daily_eta_np * cp.sqrt(252.0)

def estimate_rho(logrets_np):
    if len(logrets_np) < 2:
        return RHO_DEFAULT
    logrets_np_arr = np.array(logrets_np, dtype=np.float64)
    sq_np = logrets_np_arr ** 2
    c_np = np_covariance(logrets_np_arr, sq_np)
    var_logrets = np_variance(logrets_np_arr)
    var_sq = np_variance(sq_np)
    if var_logrets == 0 or var_sq == 0:
        return RHO_DEFAULT
    denom_np = cp.sqrt(var_logrets * var_sq)
    rho_np = c_np / denom_np if denom_np != 0.0 else 0.0
    if rho_np > 0.0: 
        rho_np = -0.3 
    return np.clip(rho_np, CLIP_RHO_MIN, CLIP_RHO_MAX)


def estimate_base_params(historical_prices_np, dt_yr_np=1/252):
    if len(historical_prices_np) < 21: 
        s0_to_use = historical_prices_np[-1] if len(historical_prices_np) > 0 else S0_DEFAULT
        return s0_to_use, XI_DEFAULT, H_DEFAULT, ETA_DEFAULT, RHO_DEFAULT

    prices_np = np.array(historical_prices_np, dtype=np.float64)
    S0_val = prices_np[-1]
    rets_np = log_returns(prices_np)
    if len(rets_np) == 0:
        return S0_val, XI_DEFAULT, H_DEFAULT, ETA_DEFAULT, RHO_DEFAULT

    xi_val = estimate_xi(rets_np, dt_yr_np)
    H_val = estimate_H(rets_np)
    eta_val = estimate_eta(rets_np, H_val)
    rho_val = estimate_rho(rets_np)
    
    final_s0 = S0_val
    final_xi = XI_DEFAULT if (not np.isfinite(xi_val) or xi_val <= 1e-6) else xi_val
    final_H = H_DEFAULT if not np.isfinite(H_val) else H_val
    final_eta = ETA_DEFAULT if (not np.isfinite(eta_val) or eta_val <= 1e-6) else eta_val
    final_rho = RHO_DEFAULT if not np.isfinite(rho_val) else rho_val
        
    return final_s0, final_xi, final_H, final_eta, final_rho

def seed_everything(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    cp.random.seed(seed_val)

def next_power_of_two(n_val):
    p_val = 1
    while p_val < n_val:
        p_val <<= 1
    return p_val

def rbergomi_lambda_gpu(time_grid_gpu, H_arr_gpu):
    return 0.5 * (time_grid_gpu[None, :] ** (2 * H_arr_gpu[:, None]))

def rbergomi_phi_gpu(lam_arr_gpu):
    N_phi_paths, N_phi_time = lam_arr_gpu.shape
    M_phi = next_power_of_two(N_phi_time)
    lam_padded_gpu = cp.zeros((N_phi_paths, M_phi), dtype=cp.float64)
    lam_padded_gpu[:, :N_phi_time] = lam_arr_gpu
    return cp.fft.fft(lam_padded_gpu, axis=1)

def fractional_gaussian_gpu(phi_val_gpu, Z_val_gpu, H_param_arr, eta_param_arr, out_len_val):
    if Z_val_gpu.ndim == 3: 
        A_gpu = phi_val_gpu[:, None, :] * Z_val_gpu
        A_ifft_gpu = cp.fft.ifft(A_gpu, axis=2).real
        scale_val = cp.sqrt(2 * H_param_arr[:, None, None]) * eta_param_arr[:, None, None]
        result = scale_val * A_ifft_gpu[..., :out_len_val]
        del A_gpu, A_ifft_gpu
        cp.cuda.Stream.null.synchronize()
        return result
    elif Z_val_gpu.ndim == 2:
        A_gpu = phi_val_gpu * Z_val_gpu 
        A_ifft_gpu = cp.fft.ifft(A_gpu, axis=1).real
        scale_val = cp.sqrt(2 * H_param_arr[:, None]) * eta_param_arr[:, None]
        return scale_val * A_ifft_gpu[..., :out_len_val]
    else:
        raise ValueError("Z_val_gpu must be 2D or 3D")


def forward_variance_gpu(X_gpu, t_grid_gpu, xi_param_arr, H_param_arr, eta_param_arr):
    v_gpu = cp.zeros_like(X_gpu)
    
    for i_val in range(X_gpu.shape[-1]):
        t_val = t_grid_gpu[i_val]
        
        if X_gpu.ndim == 2: 
            ma_gpu_path_specific = -0.5 * eta_param_arr * eta_param_arr * (t_val ** (2 * H_param_arr))
            v_gpu[:, i_val] = xi_param_arr * cp.exp(X_gpu[:, i_val] + ma_gpu_path_specific)
        elif X_gpu.ndim == 3: 
            ma_gpu_path_specific_batch = -0.5 * eta_param_arr * eta_param_arr * (t_val ** (2 * H_param_arr))
            v_gpu[..., i_val] = xi_param_arr[:, None] * cp.exp(X_gpu[..., i_val] + ma_gpu_path_specific_batch[:,None])
    return v_gpu


def price_rbergomi_option_gpu(S0_batch_gpu, K_batch_gpu, T_opt_val, r_opt_val, 
                              xi_batch_gpu, H_batch_gpu, eta_batch_gpu, rho_batch_gpu, 
                              option_type_opt, n_mc_paths_per_option, dt_opt_val):
    batch_size = S0_batch_gpu.shape[0]
    n_steps_opt = int(T_opt_val / dt_opt_val)

    if n_steps_opt <= 0:
        payoffs_gpu = cp.zeros_like(S0_batch_gpu)
        if option_type_opt == 'call':
            payoffs_gpu = cp.maximum(S0_batch_gpu - K_batch_gpu, 0.0)
        elif option_type_opt == 'put':
            payoffs_gpu = cp.maximum(K_batch_gpu - S0_batch_gpu, 0.0)
        return payoffs_gpu * cp.exp(-r_opt_val * T_opt_val)

    time_grid_opt_gpu = cp.linspace(0, n_steps_opt * dt_opt_val, n_steps_opt + 1, dtype=cp.float64)
    
    lam_opt_gpu_batch = 0.5 * (time_grid_opt_gpu[None, :] ** (2 * H_batch_gpu[:, None]))
    
    N_phi_paths_opt, N_phi_time_opt = lam_opt_gpu_batch.shape
    M_opt_val = next_power_of_two(N_phi_time_opt)
    lam_padded_opt_gpu = cp.zeros((N_phi_paths_opt, M_opt_val), dtype=cp.float64)
    lam_padded_opt_gpu[:, :N_phi_time_opt] = lam_opt_gpu_batch
    phi_opt_gpu_batch = cp.fft.fft(lam_padded_opt_gpu, axis=1)


    Z_opt_gpu = cp.random.normal(size=(batch_size, n_mc_paths_per_option, M_opt_val), dtype=cp.float64) + \
                1j * cp.random.normal(size=(batch_size, n_mc_paths_per_option, M_opt_val), dtype=cp.float64)

    X_paths_opt_gpu = fractional_gaussian_gpu(phi_opt_gpu_batch, Z_opt_gpu, H_batch_gpu, eta_batch_gpu, n_steps_opt + 1)
    
    v_paths_opt_gpu = forward_variance_gpu(X_paths_opt_gpu, time_grid_opt_gpu, xi_batch_gpu, H_batch_gpu, eta_batch_gpu)
    
    w_complex_opt_gpu = cp.fft.ifft(Z_opt_gpu, axis=2, n=M_opt_val)
    dW1_unscaled_opt_gpu = w_complex_opt_gpu.real * cp.sqrt(float(M_opt_val))
    dW2_unscaled_opt_gpu = w_complex_opt_gpu.imag * cp.sqrt(float(M_opt_val))

    current_prices_opt_gpu = cp.full((batch_size, n_mc_paths_per_option), S0_batch_gpu[:, None], dtype=cp.float64)
    sqrt_dt_opt = cp.sqrt(dt_opt_val)

    for j_opt_val in range(1, n_steps_opt + 1):
        dw1_opt_gpu = sqrt_dt_opt * dW1_unscaled_opt_gpu[..., j_opt_val - 1]
        dw2_opt_gpu = sqrt_dt_opt * dW2_unscaled_opt_gpu[..., j_opt_val - 1]
        
        dW_opt_gpu = rho_batch_gpu[:, None] * dw1_opt_gpu + cp.sqrt(cp.maximum(0.0, 1.0 - rho_batch_gpu[:, None] * rho_batch_gpu[:, None])) * dw2_opt_gpu
        
        vt_opt_gpu = v_paths_opt_gpu[..., j_opt_val - 1]
        drift_opt_gpu = (r_opt_val - 0.5 * vt_opt_gpu) * dt_opt_val
        diff_opt_gpu = cp.sqrt(cp.maximum(0.0, vt_opt_gpu)) * dW_opt_gpu
        current_prices_opt_gpu *= cp.exp(drift_opt_gpu + diff_opt_gpu)
        current_prices_opt_gpu = cp.maximum(current_prices_opt_gpu, 1e-8)

    terminal_prices_opt_gpu = current_prices_opt_gpu

    payoffs_final_gpu = cp.zeros_like(terminal_prices_opt_gpu)
    if option_type_opt == 'call':
        payoffs_final_gpu = cp.maximum(terminal_prices_opt_gpu - K_batch_gpu[:, None], 0.0)
    elif option_type_opt == 'put':
        payoffs_final_gpu = cp.maximum(K_batch_gpu[:, None] - terminal_prices_opt_gpu, 0.0)
        
    option_prices_batch_gpu = cp.mean(payoffs_final_gpu, axis=1) * cp.exp(-r_opt_val * T_opt_val)
    
    # Clean up large intermediate arrays
    del X_paths_opt_gpu, v_paths_opt_gpu, w_complex_opt_gpu
    del dW1_unscaled_opt_gpu, dW2_unscaled_opt_gpu, payoffs_final_gpu
    
    return option_prices_batch_gpu


# Generates paths and options using the rBergomi model.
def generate_paths_and_options(historical_prices_np_main, num_paths_main, r_main, dt_main, seed_main_val=None):
    start_step = 1
    paths_main_gpu = None
    v_main_gpu = None
    call_prices_atm_gpu = None
    put_prices_atm_gpu = None
    S0_arr_gpu, xi_arr_gpu, H_arr_gpu, eta_arr_gpu, rho_arr_gpu = (None,) * 5
    Z_main_gpu, dW1_unscaled_main_gpu, dW2_unscaled_main_gpu = (None,) * 3
    S0_base, xi_base, H_base, eta_base, rho_base = (None,) * 5
    total_option_pricing_time_tracker = 0.0
    loaded_seed_used_for_run = seed_main_val

    actual_checkpoint_temp_file = CHECKPOINT_FILE_TEMP_STEM + ".npz"

    try:
        if os.path.exists(CHECKPOINT_FILE):
            print(f"Loading checkpoint from {CHECKPOINT_FILE}...")
            with np.load(CHECKPOINT_FILE, allow_pickle=True) as chkpt:
                paths_main_gpu = cp.asarray(chkpt['paths_main_gpu'])
                v_main_gpu = cp.asarray(chkpt['v_main_gpu'])
                call_prices_atm_gpu = cp.asarray(chkpt['call_prices_atm_gpu'])
                put_prices_atm_gpu = cp.asarray(chkpt['put_prices_atm_gpu'])
                S0_arr_gpu = cp.asarray(chkpt['S0_arr_gpu'])
                xi_arr_gpu = cp.asarray(chkpt['xi_arr_gpu'])
                H_arr_gpu = cp.asarray(chkpt['H_arr_gpu'])
                eta_arr_gpu = cp.asarray(chkpt['eta_arr_gpu'])
                rho_arr_gpu = cp.asarray(chkpt['rho_arr_gpu'])
                Z_main_gpu = cp.asarray(chkpt['Z_main_gpu'])
                dW1_unscaled_main_gpu = cp.asarray(chkpt['dW1_unscaled_main_gpu'])
                dW2_unscaled_main_gpu = cp.asarray(chkpt['dW2_unscaled_main_gpu'])
                S0_base = chkpt['S0_base'].item()
                xi_base = chkpt['xi_base'].item()
                H_base = chkpt['H_base'].item()
                eta_base = chkpt['eta_base'].item()
                rho_base = chkpt['rho_base'].item()
                last_completed_step = chkpt['last_completed_step'].item()
                total_option_pricing_time_tracker = chkpt['total_option_pricing_time_tracker_chkpt'].item()
                loaded_seed_used_for_run = chkpt['seed_used_for_run'].item()
                start_step = last_completed_step + 1
            print(f"Resuming from end of day {last_completed_step}. Seed for this run was: {loaded_seed_used_for_run}")
        else:
            raise FileNotFoundError("No checkpoint file found.")
    except Exception as e:
        print(f"No valid checkpoint found or error loading: {e}. Starting from scratch.")
        start_step = 1
        total_option_pricing_time_tracker = 0.0
        loaded_seed_used_for_run = seed_main_val 

        if loaded_seed_used_for_run is not None:
            seed_everything(loaded_seed_used_for_run)

        S0_base, xi_base, H_base, eta_base, rho_base = estimate_base_params(historical_prices_np_main, dt_main)
        print(f"Base Estimated rBergomi Params: S0={S0_base:.2f}, xi={xi_base:.4f}, H={H_base:.4f}, eta={eta_base:.4f}, rho={rho_base:.4f}")

        S0_arr_gpu = S0_base * (1 + cp.random.normal(loc=0.0, scale=PERTURB_S0_STD, size=num_paths_main))
        xi_arr_gpu = xi_base * cp.maximum(MIN_XI_FACTOR, (1 + cp.random.normal(loc=0.0, scale=PERTURB_XI_STD, size=num_paths_main)))
        H_arr_gpu = cp.clip(H_base * (1 + cp.random.normal(loc=0.0, scale=PERTURB_H_STD, size=num_paths_main)), CLIP_H_MIN, CLIP_H_MAX)
        eta_arr_gpu = eta_base * cp.maximum(MIN_ETA_FACTOR, (1 + cp.random.normal(loc=0.0, scale=PERTURB_ETA_STD, size=num_paths_main)))
        rho_arr_gpu = cp.clip(rho_base * (1 + cp.random.normal(loc=0.0, scale=PERTURB_RHO_STD, size=num_paths_main)), CLIP_RHO_MIN, CLIP_RHO_MAX)
        
        print(f"Perturbed Param Means: S0={cp.mean(S0_arr_gpu):.2f}, xi={cp.mean(xi_arr_gpu):.4f}, H={cp.mean(H_arr_gpu):.4f}, eta={cp.mean(eta_arr_gpu):.4f}, rho={cp.mean(rho_arr_gpu):.4f}")

        time_grid_main_gpu_init = cp.linspace(0, N_STEPS * dt_main, N_STEPS + 1, dtype=cp.float64)
        lam_main_gpu_batch_init = rbergomi_lambda_gpu(time_grid_main_gpu_init, H_arr_gpu)
        
        N_phi_paths_main_init, N_phi_time_main_init = lam_main_gpu_batch_init.shape
        M_main_val_init = next_power_of_two(N_phi_time_main_init)
        
        Z_main_gpu = cp.random.normal(size=(num_paths_main, M_main_val_init), dtype=cp.float64) + \
                     1j * cp.random.normal(size=(num_paths_main, M_main_val_init), dtype=cp.float64)
        
        w_complex_main_gpu = cp.fft.ifft(Z_main_gpu, axis=1, n=M_main_val_init)
        dW1_unscaled_main_gpu = w_complex_main_gpu.real * cp.sqrt(float(M_main_val_init))
        dW2_unscaled_main_gpu = w_complex_main_gpu.imag * cp.sqrt(float(M_main_val_init))
        
        # Clean up intermediate GPU arrays
        del w_complex_main_gpu
        cp.cuda.Stream.null.synchronize()

        paths_main_gpu = cp.zeros((num_paths_main, N_STEPS + 1), dtype=cp.float64)
        paths_main_gpu[:, 0] = S0_arr_gpu
        v_main_gpu = cp.zeros((num_paths_main, N_STEPS + 1), dtype=cp.float64) 
        call_prices_atm_gpu = cp.zeros((num_paths_main, N_STEPS), dtype=cp.float64)
        put_prices_atm_gpu = cp.zeros((num_paths_main, N_STEPS), dtype=cp.float64)

    if loaded_seed_used_for_run is not None: 
        seed_everything(loaded_seed_used_for_run)
    

    time_grid_main_gpu = cp.linspace(0, N_STEPS * dt_main, N_STEPS + 1, dtype=cp.float64)
    lam_main_gpu_batch = rbergomi_lambda_gpu(time_grid_main_gpu, H_arr_gpu)
    
    N_phi_paths_main, N_phi_time_main = lam_main_gpu_batch.shape
    M_main_val = next_power_of_two(N_phi_time_main) 
    
    if start_step == 1: 
        lam_padded_main_gpu = cp.zeros((N_phi_paths_main, M_main_val), dtype=cp.float64)
        lam_padded_main_gpu[:, :N_phi_time_main] = lam_main_gpu_batch
        phi_main_gpu_batch = cp.fft.fft(lam_padded_main_gpu, axis=1)
        X_main_gpu = fractional_gaussian_gpu(phi_main_gpu_batch, Z_main_gpu[:,:M_main_val], H_arr_gpu, eta_arr_gpu, N_STEPS + 1)
        v_main_gpu_temp = forward_variance_gpu(X_main_gpu, time_grid_main_gpu, xi_arr_gpu, H_arr_gpu, eta_arr_gpu)
        v_main_gpu[:,:] = v_main_gpu_temp[:,:] 


    sqrt_dt_main = cp.sqrt(dt_main)
    
    print("Starting/Resuming main path generation and option pricing...")
    
    with tqdm(range(start_step, N_STEPS + 1), initial=start_step-1, total=N_STEPS, desc="Main Simulation Progress", unit="day") as pbar_days:
        for j_main_val in pbar_days:
            
            current_S_for_opt_gpu_full = paths_main_gpu[:, j_main_val - 1].copy()
            current_v_for_opt_gpu_full = v_main_gpu[:, j_main_val - 1].copy()
            K_atm_step_gpu_full = cp.round(current_S_for_opt_gpu_full)
            
            opt_pricing_step_start_time = time.time()

            num_mini_batches = (num_paths_main + OPTION_PRICING_MINI_BATCH_SIZE - 1) // OPTION_PRICING_MINI_BATCH_SIZE

            if T_OPTION_TENOR > 1e-6 :
                for i_mini_batch in tqdm(range(num_mini_batches), desc=f"Day {j_main_val:3d} OptPrc", leave=False, unit="mini-batch", position=1, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                    s_idx = i_mini_batch * OPTION_PRICING_MINI_BATCH_SIZE
                    e_idx = min((i_mini_batch + 1) * OPTION_PRICING_MINI_BATCH_SIZE, num_paths_main)

                    S0_mini_batch = current_S_for_opt_gpu_full[s_idx:e_idx]
                    K_mini_batch = K_atm_step_gpu_full[s_idx:e_idx]
                    v_mini_batch = current_v_for_opt_gpu_full[s_idx:e_idx]
                    
                    H_mini_batch = H_arr_gpu[s_idx:e_idx]
                    eta_mini_batch = eta_arr_gpu[s_idx:e_idx]
                    rho_mini_batch = rho_arr_gpu[s_idx:e_idx]

                    call_prices_atm_gpu[s_idx:e_idx, j_main_val - 1] = price_rbergomi_option_gpu(
                        S0_mini_batch, K_mini_batch, T_OPTION_TENOR, r_main, 
                        v_mini_batch, H_mini_batch, eta_mini_batch, rho_mini_batch, 
                        'call', N_PATHS_OPTION_MC, dt_main
                    )
                    put_prices_atm_gpu[s_idx:e_idx, j_main_val - 1] = price_rbergomi_option_gpu(
                        S0_mini_batch, K_mini_batch, T_OPTION_TENOR, r_main, 
                        v_mini_batch, H_mini_batch, eta_mini_batch, rho_mini_batch, 
                        'put', N_PATHS_OPTION_MC, dt_main
                    )
            else: 
                 call_prices_atm_gpu[:, j_main_val - 1] = cp.maximum(current_S_for_opt_gpu_full - K_atm_step_gpu_full, 0.0)
                 put_prices_atm_gpu[:, j_main_val - 1] = cp.maximum(K_atm_step_gpu_full - current_S_for_opt_gpu_full, 0.0)
            
            opt_pricing_step_end_time = time.time()
            total_option_pricing_time_tracker += (opt_pricing_step_end_time - opt_pricing_step_start_time)
            
            dw1_main_gpu_step = sqrt_dt_main * dW1_unscaled_main_gpu[:, j_main_val - 1]
            dw2_main_gpu_step = sqrt_dt_main * dW2_unscaled_main_gpu[:, j_main_val - 1]
            
            dW_main_gpu = rho_arr_gpu * dw1_main_gpu_step + cp.sqrt(cp.maximum(0.0, 1.0 - rho_arr_gpu * rho_arr_gpu)) * dw2_main_gpu_step
            
            vt_main_gpu_step = v_main_gpu[:, j_main_val - 1]
            drift_main_gpu = (r_main - 0.5 * vt_main_gpu_step) * dt_main
            diff_main_gpu = cp.sqrt(cp.maximum(0.0, vt_main_gpu_step)) * dW_main_gpu
            
            paths_main_gpu[:, j_main_val] = paths_main_gpu[:, j_main_val - 1] * cp.exp(drift_main_gpu + diff_main_gpu)
            paths_main_gpu[:, j_main_val] = cp.maximum(paths_main_gpu[:, j_main_val], 1e-8)
            
            cp.cuda.Stream.null.synchronize()
            
            np.savez_compressed(CHECKPOINT_FILE_TEMP_STEM,
                paths_main_gpu=cp.asnumpy(paths_main_gpu),
                v_main_gpu=cp.asnumpy(v_main_gpu),
                call_prices_atm_gpu=cp.asnumpy(call_prices_atm_gpu),
                put_prices_atm_gpu=cp.asnumpy(put_prices_atm_gpu),
                S0_arr_gpu=cp.asnumpy(S0_arr_gpu), 
                xi_arr_gpu=cp.asnumpy(xi_arr_gpu),
                H_arr_gpu=cp.asnumpy(H_arr_gpu),
                eta_arr_gpu=cp.asnumpy(eta_arr_gpu),
                rho_arr_gpu=cp.asnumpy(rho_arr_gpu),
                Z_main_gpu=cp.asnumpy(Z_main_gpu), 
                dW1_unscaled_main_gpu=cp.asnumpy(dW1_unscaled_main_gpu),
                dW2_unscaled_main_gpu=cp.asnumpy(dW2_unscaled_main_gpu),
                S0_base=S0_base, xi_base=xi_base, H_base=H_base, eta_base=eta_base, rho_base=rho_base,
                last_completed_step=j_main_val,
                total_option_pricing_time_tracker_chkpt=total_option_pricing_time_tracker,
                seed_used_for_run=loaded_seed_used_for_run 
            )
            os.replace(CHECKPOINT_FILE_TEMP_STEM + ".npz", CHECKPOINT_FILE)

    print(f"Total time spent in batched option pricing calls: {total_option_pricing_time_tracker:.2f} seconds.")
    
    if start_step > N_STEPS :
        if os.path.exists(CHECKPOINT_FILE):
            try:
                os.remove(CHECKPOINT_FILE)
                print(f"Successfully removed checkpoint file: {CHECKPOINT_FILE}")
            except Exception as e:
                print(f"Error removing checkpoint file {CHECKPOINT_FILE}: {e}")
            
    return paths_main_gpu, v_main_gpu, call_prices_atm_gpu, put_prices_atm_gpu

def main():
    start_total_time = time.time()
    
    try:
        prices_np_hist = np.loadtxt(INPUT_FILE, dtype=np.float64, delimiter=',')
    except FileNotFoundError:
        print(f"Historical prices file {INPUT_FILE} not found. Using default S0 and rBergomi parameters.")
        prices_np_hist = np.array([]) 
    except Exception as e:
        print(f"Error loading historical prices: {e}. Using default S0 and rBergomi parameters.")
        prices_np_hist = np.array([])

    if prices_np_hist.ndim == 0 and prices_np_hist.size == 1:
        prices_np_hist = np.array([float(prices_np_hist)])
    elif prices_np_hist.ndim > 1:
        print("Historical prices file has more than one column, using the first column.")
        prices_np_hist = prices_np_hist[:,0]
    
    paths_gpu, vol_gpu, calls_gpu, puts_gpu = generate_paths_and_options(
        prices_np_hist, N_PATHS, R, DT, SEED
    )
    
    paths_np = cp.asnumpy(paths_gpu)
    vol_np = cp.asnumpy(vol_gpu)
    calls_np = cp.asnumpy(calls_gpu)
    puts_np = cp.asnumpy(puts_gpu)
    
    np.savez_compressed(OUTPUT_FILE, paths=paths_np, volatilities=vol_np, call_prices_atm=calls_np, put_prices_atm=puts_np)
    print(f"Saved {N_PATHS} paths with {N_STEPS} steps, along with volatilities and ATM option prices to {OUTPUT_FILE}")
    
    end_total_time = time.time()
    print(f"Total script execution time: {(end_total_time - start_total_time):.2f} seconds.")

if __name__ == '__main__':
    main()
