import numpy as np
from scipy.stats import norm
from typing import Dict

class OptionCalculator:
    """Black-Scholes option pricing and Greeks calculator"""

    def __init__(self):
        pass

    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price"""
        SIGMA_FLOOR = 1e-4      
        TIME_FLOOR  = 1e-4   
        PRICE_FLOOR = 1e-6    
        sigma = max(sigma, SIGMA_FLOOR)
        T     = max(T, TIME_FLOOR)
        S     = max(S,  PRICE_FLOOR)
        K     = max(K,  PRICE_FLOOR)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        CLIP = 10.0            
        d1 = np.clip(d1, -CLIP, CLIP)
        d2 = np.clip(d2, -CLIP, CLIP)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        if not np.isfinite(price):
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        return max(price, 0)

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks (delta, gamma, vega)"""
        SIGMA_FLOOR = 1e-4      
        TIME_FLOOR  = 1e-4      
        PRICE_FLOOR = 1e-6      
        sigma = max(sigma, SIGMA_FLOOR)
        T     = max(T, TIME_FLOOR)
        S     = max(S,  PRICE_FLOOR)
        K     = max(K,  PRICE_FLOOR)
        
    
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        CLIP = 10.0            
        d1 = np.clip(d1, -CLIP, CLIP)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  
            delta = norm.cdf(d1) - 1.0
        
        denom = S * sigma * np.sqrt(T)
        gamma = norm.pdf(d1) / denom if denom > 1e-10 else 0.0
        
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        price_bs = self.black_scholes_price(S, K, T, r, sigma, option_type)
        return_dict = {
            'price': price_bs,
            'delta': delta,
            'gamma': gamma,
            'vega': vega
        }

        for k, v in list(return_dict.items()):
            if not np.isfinite(v):
                return_dict[k] = 0.0
        
        return return_dict
