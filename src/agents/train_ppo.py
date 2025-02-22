# Basic PPO training pipeline for hedging environment

import os
import sys
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from env.hedging_env import HedgingEnv

# Basic configuration
DATA_FILE = "./data/paths_rbergomi_options_100k.npz"
N_ENVS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_env(seed):
    """Create a monitored environment"""
    def _init():
        env = HedgingEnv(
            data_file_path=DATA_FILE,
            loss_type="abs",
            pnl_penalty_weight=0.01,
            lambda_cost=1.0
        )
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

def train_agent(loss_type="abs", pnl_penalty_weight=0.01, lambda_cost=1.0, 
                total_timesteps=100000, seed=42):
    """Basic PPO training function"""
    
    print(f"Starting PPO training with loss_type={loss_type}, w={pnl_penalty_weight}, lambda={lambda_cost}")
    print(f"Device: {DEVICE}")
    
    # Set seeds
    set_random_seed(seed)
    
    # Create vectorized environment
    env_fns = [create_env(seed + i) for i in range(N_ENVS)]
    env = SubprocVecEnv(env_fns)
    
    # Basic PPO hyperparameters (simple defaults)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=DEVICE,
        verbose=1
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model_path = f"ppo_hedge_{loss_type}_w{pnl_penalty_weight}_l{lambda_cost}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()
    return model

def main():
    parser = argparse.ArgumentParser(description="Basic PPO training for hedging")
    parser.add_argument("--loss_type", type=str, default="abs", help="Loss type")
    parser.add_argument("--w", type=float, default=0.01, help="PnL penalty weight") 
    parser.add_argument("--lambda", type=float, default=1.0, help="Transaction cost weight")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # TODO: add hyperparameter optimization
    # TODO: add checkpointing for long runs
    # TODO: add evaluation pipeline
    
    train_agent(
        loss_type=args.loss_type,
        pnl_penalty_weight=args.w,
        lambda_cost=getattr(args, 'lambda'),
        total_timesteps=args.timesteps,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
