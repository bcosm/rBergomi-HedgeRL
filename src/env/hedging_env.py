# Environment for RL-based hedging simulations.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import norm
import time

class HedgingEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 1}

    # Initializes the hedging environment with specified parameters.
    def __init__(self, data_file_path,
                 transaction_cost_per_contract=0.65,
                 lambda_cost=1.0,
                 pnl_penalty_weight=0.01,
                 loss_type="abs", 
                 initial_cash=0.0,
                 shares_to_hedge=10000,
                 max_contracts_held_per_type=200,
                 max_trade_per_step=100, 
                 profile_print_interval=0,
                 record_metrics=True):

        super().__init__()

        self.pnl_penalty_weight = pnl_penalty_weight
        self.lambda_cost = lambda_cost
        self.loss_type = loss_type
        self.record_metrics = record_metrics
        self._max_trade_per_step_internal = max_trade_per_step


        try:
            data = np.load(data_file_path)
            self.stock_paths = data['paths'].astype(np.float32)
            self.vol_paths = data['volatilities'].astype(np.float32)
            self.call_prices_paths = data['call_prices_atm'].astype(np.float32)
            self.put_prices_paths = data['put_prices_atm'].astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(f"Could not load or parse data from {data_file_path}. Error: {e}")

        if not (self.stock_paths.shape == self.vol_paths.shape and \
                self.stock_paths.shape[0] == self.call_prices_paths.shape[0] == self.put_prices_paths.shape[0] and \
                self.stock_paths.shape[1] == self.call_prices_paths.shape[1] + 1 == self.put_prices_paths.shape[1] + 1):
            raise ValueError("Data shapes are inconsistent.")

        self.num_episodes = self.stock_paths.shape[0]
        self.episode_length = self.stock_paths.shape[1] - 1

        self.transaction_cost_per_contract = transaction_cost_per_contract
        self.initial_cash = initial_cash
        self.max_contracts_held = max_contracts_held_per_type
        self.shares_held_fixed = shares_to_hedge
        self.option_contract_multiplier = 100
        self.risk_free_rate = 0.04
        self.option_tenor_years = 30 / 252

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        low_bounds = np.array([
            0.1, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0
        ], dtype=np.float32)
        high_bounds = np.array([
            10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 1.0, 50.0, 1.0, 1.0
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(13,), dtype=np.float32)

        self.current_episode_idx = -1
        self.current_step = 0
        self.initial_S0_for_episode = 1.0

        self._profile_step_call_count = 0
        self._profile_get_obs_call_count = 0
        self._profile_calc_greeks_call_count = 0
        self._profile_print_interval = profile_print_interval

    def _calculate_greeks(self, S, K, T, r, v_spot):
        if not self.record_metrics:
            return 0.0, 0.0, 0.0, 0.0

        self._profile_calc_greeks_call_count += 1
        sigma = np.sqrt(np.maximum(v_spot, 1e-8))
        call_delta, put_delta, gamma = 0.0, 0.0, 0.0

        if S <= 1e-6:
            call_delta = 0.5 if K == 0 else (0.0 if K > 0 else 1.0)
            put_delta = -0.5 if K == 0 else (0.0 if K < 0 else -1.0)
        elif T <= 1e-6 or sigma <= 1e-6:
            call_delta = 1.0 if S > K else (0.5 if S == K else 0.0)
            put_delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        else:
            K_checked = np.maximum(K, 1e-6)
            sigma_sqrt_T = sigma * np.sqrt(T)
            if sigma_sqrt_T < 1e-9:
                d1 = np.sign(np.log(S / K_checked) + (r + 0.5 * sigma**2) * T) * 10.0
            else:
                d1 = (np.log(S / K_checked) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
            call_delta = norm.cdf(d1)
            put_delta = norm.cdf(d1) - 1.0
            gamma_denominator = S * sigma_sqrt_T
            if abs(gamma_denominator) < 1e-9:
                gamma = 0.0
            else:
                gamma = norm.pdf(d1) / gamma_denominator
        return call_delta, gamma, put_delta, gamma

    def _get_observation(self):
        self._profile_get_obs_call_count += 1
        S_t = self.current_stock_price
        C_t = self.current_call_price
        P_t = self.current_put_price
        v_t = self.current_volatility

        s0_safe_obs = np.maximum(self.initial_S0_for_episode, 25.0) 
        norm_S_t = S_t / s0_safe_obs
        norm_C_t = C_t / s0_safe_obs
        norm_P_t = P_t / s0_safe_obs
        norm_call_held = self.call_contracts_held / self.max_contracts_held if self.max_contracts_held != 0 else 0.0
        norm_put_held = self.put_contracts_held / self.max_contracts_held if self.max_contracts_held != 0 else 0.0
        norm_time_to_end = (self.episode_length - self.current_step) / self.episode_length if self.episode_length != 0 else 0.0

        K_atm_t = np.round(S_t)
        call_delta, call_gamma, put_delta, put_gamma = self._calculate_greeks(
            S_t, K_atm_t, self.option_tenor_years, self.risk_free_rate, v_t
        )

        if self.current_step == 0 or self.S_t_minus_1 == 0:
            lagged_S_return = 0.0
            lagged_v_change = 0.0
        else:
            lagged_S_return = (S_t - self.S_t_minus_1) / self.S_t_minus_1
            lagged_v_change = v_t - self.v_t_minus_1
        lagged_S_return = np.clip(lagged_S_return, -1.0, 1.0)
        lagged_v_change = np.clip(lagged_v_change, -1.0, 1.0)

        obs = np.array([
            norm_S_t, norm_C_t, norm_P_t, norm_call_held, norm_put_held,
            v_t, norm_time_to_end, call_delta, call_gamma,
            put_delta, put_gamma, lagged_S_return, lagged_v_change
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.current_episode_idx = self.np_random.integers(self.num_episodes)
        self.current_path_S = self.stock_paths[self.current_episode_idx]
        self.current_path_v = self.vol_paths[self.current_episode_idx]
        self.current_path_C = self.call_prices_paths[self.current_episode_idx]
        self.current_path_P = self.put_prices_paths[self.current_episode_idx]
        self.current_step = 0
        self.initial_S0_for_episode = self.current_path_S[0]
        if self.initial_S0_for_episode < 1e-6 : self.initial_S0_for_episode = 1.0 
        
        self.current_stock_price = self.current_path_S[self.current_step]
        self.current_volatility = self.current_path_v[self.current_step]
        self.current_call_price = self.current_path_C[self.current_step]
        self.current_put_price = self.current_path_P[self.current_step]
        self.call_contracts_held = 0
        self.put_contracts_held = 0
        self.cash_balance = self.initial_cash
        initial_options_value = 0
        self.portfolio_value_t_minus_1 = (self.shares_held_fixed * self.current_stock_price) + \
                                         initial_options_value + self.cash_balance
        self.S_t_minus_1 = self.current_stock_price
        self.v_t_minus_1 = self.current_volatility
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        self._profile_step_call_count +=1
        
        raw_call_action = action[0] 
        raw_put_action = action[1]  

        contracts_float_call = raw_call_action * self._max_trade_per_step_internal
        contracts_float_put = raw_put_action * self._max_trade_per_step_internal
        
        requested_call_trade_rounded = np.rint(contracts_float_call).astype(int)
        requested_put_trade_rounded = np.rint(contracts_float_put).astype(int)
        
        requested_call_trade_clipped = np.clip(requested_call_trade_rounded, -self._max_trade_per_step_internal, self._max_trade_per_step_internal)
        requested_put_trade_clipped = np.clip(requested_put_trade_rounded, -self._max_trade_per_step_internal, self._max_trade_per_step_internal)

        prev_call_contracts = self.call_contracts_held
        prev_put_contracts = self.put_contracts_held

        potential_call_contracts = self.call_contracts_held + requested_call_trade_clipped
        potential_put_contracts = self.put_contracts_held + requested_put_trade_clipped

        self.call_contracts_held = np.clip(potential_call_contracts, -self.max_contracts_held, self.max_contracts_held).astype(int)
        self.put_contracts_held = np.clip(potential_put_contracts, -self.max_contracts_held, self.max_contracts_held).astype(int)

        actual_calls_traded_this_step = self.call_contracts_held - prev_call_contracts
        actual_puts_traded_this_step = self.put_contracts_held - prev_put_contracts

        transaction_costs_this_step = (abs(actual_calls_traded_this_step) + abs(actual_puts_traded_this_step)) * \
                                      self.transaction_cost_per_contract
        self.cash_balance -= transaction_costs_this_step

        self.S_t_minus_1 = self.current_stock_price
        self.v_t_minus_1 = self.current_volatility

        self.current_step += 1
        terminated = (self.current_step >= self.episode_length)
        truncated = False

        self.current_stock_price = self.current_path_S[self.current_step]
        self.current_volatility = self.current_path_v[self.current_step]

        if not terminated:
            self.current_call_price = self.current_path_C[self.current_step]
            self.current_put_price = self.current_path_P[self.current_step]
        else:
            self.current_call_price = self.current_path_C[self.current_step -1]
            self.current_put_price = self.current_path_P[self.current_step -1]

        current_options_value = (self.call_contracts_held * self.current_call_price * self.option_contract_multiplier) + \
                                (self.put_contracts_held * self.current_put_price * self.option_contract_multiplier)
        portfolio_value_t_plus_1 = (self.shares_held_fixed * self.current_stock_price) + \
                                   current_options_value + self.cash_balance
        step_pnl = portfolio_value_t_plus_1 - self.portfolio_value_t_minus_1
        per_share_step_pnl = step_pnl / self.shares_held_fixed if self.shares_held_fixed != 0 else step_pnl
        
        raw_pnl_deviation_abs_value = np.abs(per_share_step_pnl)
        
        pnl_term_value = 0.0
        s0_floor = np.maximum(self.initial_S0_for_episode, 25.0) 

        if self.loss_type == "mse":
            pnl_term_value = (per_share_step_pnl**2) / (s0_floor**2 + 1e-9)
        elif self.loss_type == "abs":
            pnl_term_value = np.abs(per_share_step_pnl) / (s0_floor + 1e-9)
        elif self.loss_type == "cvar": 
            pnl_term_value = np.abs(per_share_step_pnl) / (s0_floor + 1e-9)
        else: 
            pnl_term_value = np.abs(per_share_step_pnl) / (s0_floor + 1e-9)

        reward_pnl_component = -self.pnl_penalty_weight * pnl_term_value
        transaction_cost_penalty = self.lambda_cost * transaction_costs_this_step
        current_reward = reward_pnl_component - transaction_cost_penalty
        
        self.portfolio_value_t_minus_1 = portfolio_value_t_plus_1
        observation = self._get_observation()

        info = {
            'step_pnl_total': step_pnl,
            'per_share_step_pnl': per_share_step_pnl, 
            'raw_pnl_deviation_abs': raw_pnl_deviation_abs_value,
            'transaction_costs_total': transaction_costs_this_step,
            'reward_pnl_component': reward_pnl_component,
            'transaction_cost_penalty': transaction_cost_penalty,
            'reward_step': current_reward,
            'portfolio_value': portfolio_value_t_plus_1,
            'call_contracts': self.call_contracts_held,
            'put_contracts': self.put_contracts_held,
            'cash': self.cash_balance,
            'raw_action_call': raw_call_action,
            'raw_action_put': raw_put_action,
            'scaled_float_call': contracts_float_call,
            'scaled_float_put': contracts_float_put,
            'requested_calls_rounded_clipped': requested_call_trade_clipped,
            'requested_puts_rounded_clipped': requested_put_trade_clipped,
            'actual_calls_traded': actual_calls_traded_this_step,
            'actual_puts_traded': actual_puts_traded_this_step,
            'loss_type_used': self.loss_type,
            'initial_S0_for_episode': self.initial_S0_for_episode 
        }
        return observation, current_reward, terminated, truncated, info

    def render(self):
        return None
        
    def close(self):
        return None
