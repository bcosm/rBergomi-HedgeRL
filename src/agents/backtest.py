# Script for backtesting RL-based hedging strategies.

import os, torch, gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pathlib
import sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.env.hedging_env import HedgingEnv

MODEL_DIR     = "src/results/models/lossabs_w0.0001_l0.01"
model_path    = os.path.join(MODEL_DIR, "final_model.zip")
vecnorm_path  = os.path.join(MODEL_DIR, "final_vecnormalize.pkl")
paths_file    = "data/paths_rbergomi_options_100k.npz"

def make_env(seed=0):
    def _init():
        env = HedgingEnv(
            data_file_path=paths_file,
            loss_type="abs", pnl_penalty_weight=1e-4, lambda_cost=0.01,
            record_metrics=True
        )
        env.reset(seed=seed)
        return env
    return _init

raw_vec_env = DummyVecEnv([make_env(12345)])

eval_env = VecNormalize.load(vecnorm_path, raw_vec_env)
eval_env.training = False
eval_env.norm_obs = True
eval_env.norm_reward = False

model = RecurrentPPO.load(model_path, env=eval_env, device="cpu")

from stable_baselines3.common.evaluation import evaluate_policy
mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
print(f"Mean episode reward: {mean_r:.3f} ± {std_r:.3f}")
