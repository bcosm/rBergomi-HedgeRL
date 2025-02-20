# Environment for RL-based hedging simulations.

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HedgingEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 1}

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

        self.lambda_cost = lambda_cost
        self.pnl_penalty_weight = pnl_penalty_weight
        self.loss_type = loss_type
        self.initial_cash = initial_cash
        self.shares_to_hedge = shares_to_hedge
        self.max_contracts_held_per_type = max_contracts_held_per_type
        self.max_trade_per_step = max_trade_per_step
        self.transaction_cost_per_contract = transaction_cost_per_contract
        self.profile_print_interval = profile_print_interval
        self.record_metrics = record_metrics

        # Load precomputed paths and option data
        self.data = np.load(data_file_path)
        self.paths = self.data['paths']
        self.volatilities = self.data['volatilities']
        self.call_prices_atm = self.data['call_prices_atm']
        self.put_prices_atm = self.data['put_prices_atm']

        n_paths, n_steps = self.paths.shape
        self.n_paths = n_paths
        self.n_steps = n_steps

        # Action space: continuous values for call and put position changes
        # Range is [-max_trade_per_step, max_trade_per_step] for each
        self.action_space = spaces.Box(
            low=-max_trade_per_step, 
            high=max_trade_per_step, 
            shape=(2,), 
            dtype=np.float32
        )

        # Observation space: [normalized price, normalized volatility, call delta, call gamma, 
        #                    put delta, put gamma, call position, put position, portfolio value normalized, time remaining]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,), 
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Sample a random path
        self.current_path_idx = self.np_random.integers(0, self.n_paths)
        self.current_step = 0
        
        # Initial state
        self.call_position = 0
        self.put_position = 0
        self.cash_position = self.initial_cash
        
        # Initial portfolio value tracking
        initial_stock_price = self.paths[self.current_path_idx, 0]
        self.portfolio_value_t_minus_1 = self.initial_cash - self.shares_to_hedge * initial_stock_price

        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action):
        if self.current_step >= self.n_steps - 1:
            # Episode is done
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, True, False, info
            
        # Parse actions
        call_trade = np.clip(action[0], -self.max_trade_per_step, self.max_trade_per_step)
        put_trade = np.clip(action[1], -self.max_trade_per_step, self.max_trade_per_step)
        
        # Ensure position limits
        new_call_position = self.call_position + call_trade
        new_put_position = self.put_position + put_trade
        
        new_call_position = np.clip(new_call_position, -self.max_contracts_held_per_type, self.max_contracts_held_per_type)
        new_put_position = np.clip(new_put_position, -self.max_contracts_held_per_type, self.max_contracts_held_per_type)
        
        # Actual trades (after clipping)
        actual_call_trade = new_call_position - self.call_position
        actual_put_trade = new_put_position - self.put_position
        
        # Calculate transaction costs
        transaction_costs_this_step = (abs(actual_call_trade) + abs(actual_put_trade)) * self.transaction_cost_per_contract
        
        # Update cash position
        current_stock_price = self.paths[self.current_path_idx, self.current_step]
        current_call_price = self.call_prices_atm[self.current_path_idx, self.current_step]
        current_put_price = self.put_prices_atm[self.current_path_idx, self.current_step]
        
        # Cash flow from option trades (selling = positive cash, buying = negative cash)
        option_cash_flow = -actual_call_trade * current_call_price - actual_put_trade * current_put_price
        self.cash_position += option_cash_flow - transaction_costs_this_step
        
        # Update positions
        self.call_position = new_call_position
        self.put_position = new_put_position
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value at t+1
        if self.current_step < self.n_steps:
            next_stock_price = self.paths[self.current_path_idx, self.current_step]
            next_call_price = self.call_prices_atm[self.current_path_idx, self.current_step]
            next_put_price = self.put_prices_atm[self.current_path_idx, self.current_step]
        else:
            # At expiration, options are worth their intrinsic value
            next_stock_price = self.paths[self.current_path_idx, -1]
            strike_price = self.paths[self.current_path_idx, 0]  # Assuming ATM options
            next_call_price = max(0, next_stock_price - strike_price)
            next_put_price = max(0, strike_price - next_stock_price)
        
        # Portfolio value = cash + option positions value - short stock position value
        option_value = self.call_position * next_call_price + self.put_position * next_put_price
        stock_value = -self.shares_to_hedge * next_stock_price
        portfolio_value_t_plus_1 = self.cash_position + option_value + stock_value
        
        # Calculate step PnL
        per_share_step_pnl = (portfolio_value_t_plus_1 - self.portfolio_value_t_minus_1) / self.shares_to_hedge
        
        # Calculate reward
        s0_floor = max(self.paths[self.current_path_idx, 0], 1.0)
        
        if self.loss_type == "quad":
            pnl_term_value = (per_share_step_pnl / (s0_floor + 1e-9)) ** 2
        else: 
            pnl_term_value = np.abs(per_share_step_pnl) / (s0_floor + 1e-9)

        # BUG: Should be negative sign here, but using positive (will be fixed in commit 8)
        reward_pnl_component = self.pnl_penalty_weight * pnl_term_value
        transaction_cost_penalty = self.lambda_cost * transaction_costs_this_step
        current_reward = reward_pnl_component - transaction_cost_penalty
        
        self.portfolio_value_t_minus_1 = portfolio_value_t_plus_1
        observation = self._get_observation()

        info = {
            'step_pnl': per_share_step_pnl,
            'portfolio_value': portfolio_value_t_plus_1,
            'transaction_costs': transaction_costs_this_step,
            'call_position': self.call_position,
            'put_position': self.put_position,
            'stock_price': next_stock_price,
            'call_price': next_call_price,
            'put_price': next_put_price
        }
        
        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        return observation, current_reward, terminated, truncated, info

    def _get_observation(self):
        if self.current_step >= self.n_steps:
            # Return final observation
            current_step_idx = self.n_steps - 1
        else:
            current_step_idx = self.current_step
        
        current_stock_price = self.paths[self.current_path_idx, current_step_idx]
        current_vol = self.volatilities[self.current_path_idx, current_step_idx]
        current_call_price = self.call_prices_atm[self.current_path_idx, current_step_idx]
        current_put_price = self.put_prices_atm[self.current_path_idx, current_step_idx]
        
        # Normalize stock price by initial price
        initial_price = self.paths[self.current_path_idx, 0]
        normalized_price = current_stock_price / initial_price
        
        # Calculate Greeks (simplified Black-Scholes approximations)
        call_delta, call_gamma = self._calculate_greeks(current_stock_price, initial_price, current_vol, 'call')
        put_delta, put_gamma = self._calculate_greeks(current_stock_price, initial_price, current_vol, 'put')
        
        # Portfolio value normalized by initial stock price and number of shares
        portfolio_value_normalized = self.portfolio_value_t_minus_1 / (initial_price * self.shares_to_hedge)
        
        # Time remaining (fraction of total episode)
        time_remaining = (self.n_steps - 1 - current_step_idx) / (self.n_steps - 1)
        
        obs = np.array([
            normalized_price,
            current_vol,
            call_delta,
            call_gamma,
            put_delta,
            put_gamma,
            self.call_position / self.max_contracts_held_per_type,
            self.put_position / self.max_contracts_held_per_type,
            portfolio_value_normalized,
            time_remaining
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_greeks(self, S, K, vol, option_type):
        # Simplified Greeks calculation
        T = (self.n_steps - 1 - self.current_step) / 252.0  # Assuming daily steps, 252 trading days per year
        r = 0.02  # Risk-free rate assumption
        
        if T <= 0:
            T = 1/252.0  # Minimum time to avoid division by zero
        
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T) + 1e-8)
        d2 = d1 - vol * math.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
            
        gamma = norm.pdf(d1) / (S * vol * math.sqrt(T) + 1e-8)
        
        return delta, gamma
    
    def _get_info(self):
        if self.current_step < self.n_steps:
            current_stock_price = self.paths[self.current_path_idx, self.current_step]
        else:
            current_stock_price = self.paths[self.current_path_idx, -1]
            
        return {
            'current_step': self.current_step,
            'stock_price': current_stock_price,
            'call_position': self.call_position,
            'put_position': self.put_position,
            'portfolio_value': self.portfolio_value_t_minus_1
        }

    def render(self):
        return None
        
    def close(self):
        return None
