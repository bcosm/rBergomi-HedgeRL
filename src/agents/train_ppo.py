# PPO training pipeline with Optuna hyperparameter optimization

import os
import sys
import argparse
import argparse
import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from env.hedging_env import HedgingEnv

# Basic configuration
# Basic configuration
DATA_FILE = "./data/paths_rbergomi_options_100k.npz"
N_ENVS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HPO configuration
N_TRIALS = 50
N_TIMESTEPS_HPO = 50000
N_EVAL_EPISODES = 10
EVAL_FREQ = 5000

# Checkpoint configuration
CHECKPOINT_FREQ = 10000
CHECKPOINT_DIR = "./checkpoints/"
BEST_MODEL_DIR = "./best_models/"

def create_env(loss_type="abs", pnl_penalty_weight=0.01, lambda_cost=1.0, seed=42):
    """Create a monitored environment"""
    def _init():
        env = HedgingEnv(
            data_file_path=DATA_FILE,
            loss_type=loss_type,
            pnl_penalty_weight=pnl_penalty_weight,
            lambda_cost=lambda_cost
        )
        env = Monitor(env)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    set_random_seed(seed)
    return _init

def train_agent(loss_type="abs", pnl_penalty_weight=0.01, lambda_cost=1.0, 
                total_timesteps=100000, seed=42, hpo_params=None):
    """PPO training function with optional hyperparameters"""
    
    print(f"Starting PPO training with loss_type={loss_type}, w={pnl_penalty_weight}, lambda={lambda_cost}")
    print(f"Device: {DEVICE}")
    
    # Set seeds
    set_random_seed(seed)
    
    # Create vectorized environment
    env_fns = [create_env(loss_type, pnl_penalty_weight, lambda_cost, seed + i) for i in range(N_ENVS)]
    env = SubprocVecEnv(env_fns)
    
    # Use HPO params if provided, otherwise defaults
    if hpo_params is None:
        hpo_params = {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5
        }
    
    # Create PPO model with hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hpo_params["learning_rate"],
        n_steps=hpo_params["n_steps"],
        batch_size=hpo_params["batch_size"],
        n_epochs=hpo_params["n_epochs"],
        gamma=hpo_params["gamma"],
        gae_lambda=hpo_params["gae_lambda"],
        clip_range=hpo_params["clip_range"],
        ent_coef=hpo_params["ent_coef"],
        vf_coef=hpo_params["vf_coef"],
        max_grad_norm=hpo_params["max_grad_norm"],
        device=DEVICE,
        verbose=0
    )
    
    # Setup checkpointing and evaluation
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix=f"rl_model_{loss_type}_w{pnl_penalty_weight}_l{lambda_cost}"
    )
    
    # Create evaluation environment for callback
    eval_env_fns = [create_env(loss_type, pnl_penalty_weight, lambda_cost, seed + 1000 + i) for i in range(N_ENVS)]
    eval_env = SubprocVecEnv(eval_env_fns)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=f"./logs/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    eval_env.close()
    
    env.close()
    return model

def objective(trial, loss_type="abs", pnl_penalty_weight=0.01, lambda_cost=1.0, seed=42):
    """Optuna objective function for hyperparameter optimization"""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1.0)
    
    hpo_params = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm
    }
    
    try:
        # Train model with sampled hyperparameters
        model = train_agent(
            loss_type=loss_type,
            pnl_penalty_weight=pnl_penalty_weight,
            lambda_cost=lambda_cost,
            total_timesteps=N_TIMESTEPS_HPO,
            seed=seed,
            hpo_params=hpo_params
        )
        
        # Evaluate model
        eval_env_fns = [create_env(loss_type, pnl_penalty_weight, lambda_cost, seed + 1000 + i) for i in range(N_ENVS)]
        eval_env = SubprocVecEnv(eval_env_fns)
        
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)
        eval_env.close()
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise TrialPruned()

def run_hyperparameter_optimization(loss_type="abs", pnl_penalty_weight=0.01, lambda_cost=1.0, seed=42):
    """Run Optuna hyperparameter optimization"""
    
    print(f"Starting hyperparameter optimization for loss_type={loss_type}, w={pnl_penalty_weight}, lambda={lambda_cost}")
    
    # Create study with median pruner
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, loss_type, pnl_penalty_weight, lambda_cost, seed),
        n_trials=N_TRIALS
    )
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params

def main():
    parser = argparse.ArgumentParser(description="PPO training with Optuna HPO for hedging")
    parser.add_argument("--mode", type=str, choices=["train", "hpo"], default="train", help="Training mode")
    parser.add_argument("--loss_type", type=str, default="abs", help="Loss type")
    parser.add_argument("--w", type=float, default=0.01, help="PnL penalty weight") 
    parser.add_argument("--lambda", type=float, default=1.0, help="Transaction cost weight")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.mode == "hpo":
        # Run hyperparameter optimization
        best_params = run_hyperparameter_optimization(
            loss_type=args.loss_type,
            pnl_penalty_weight=args.w,
            lambda_cost=getattr(args, 'lambda'),
            seed=args.seed
        )
        
        # Train final model with best parameters
        model = train_agent(
            loss_type=args.loss_type,
            pnl_penalty_weight=args.w,
            lambda_cost=getattr(args, 'lambda'),
            total_timesteps=args.timesteps,
            seed=args.seed,
            hpo_params=best_params
        )
        
        # Save the optimized model
        model_path = f"ppo_hedge_optimized_{args.loss_type}_w{args.w}_l{getattr(args, 'lambda')}.zip"
        model.save(model_path)
        print(f"Optimized model saved to {model_path}")
        
    else:
        # Basic training mode
        model = train_agent(
            loss_type=args.loss_type,
            pnl_penalty_weight=args.w,
            lambda_cost=getattr(args, 'lambda'),
            total_timesteps=args.timesteps,
            seed=args.seed
        )
        
        # Save the model
        model_path = f"ppo_hedge_{args.loss_type}_w{args.w}_l{getattr(args, 'lambda')}.zip"
        model.save(model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
