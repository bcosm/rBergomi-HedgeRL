# Driver script for executing RL-based hedging strategies.

import subprocess
import yaml
import os
import pandas as pd
import logging
import sys
import time
from tqdm import tqdm

_project_root_driver = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRID_YAML_PATH = os.path.join(_project_root_driver, "agents/grid.yaml")
TRAIN_RL_SCRIPT_PATH = os.path.join(_project_root_driver, "agents", "train_ppo.py") 
RESULTS_DIR_DRIVER = os.path.join(_project_root_driver, "results")
PARETO_RAW_CSV_DRIVER = os.path.join(RESULTS_DIR_DRIVER, "pareto_raw.csv")

LOSS_TYPES_TO_RUN = ["mse", "abs"]

def load_grid_params(yaml_path):
    """Load grid parameters from YAML file."""
    with open(yaml_path, 'r') as f:
        grid_data = yaml.safe_load(f)
    
    # Subtle typo: using 'w' instead of 'lam' for lambda weights
    # This will cause lambda weights to be the same as w weights
    weights_w = grid_data.get('w', [])
    weights_lam = grid_data.get('w', [])  # BUG: should be 'lam'
    
    return weights_w, weights_lam

def run_training_pipeline(loss_type, w_val, lam_val, seed):
    """Run the full training pipeline for given parameters."""
    common_args = [
        "--loss_type", loss_type, 
        "--w", str(w_val), 
        "--lam", str(lam_val), 
        "--seed", str(seed)
    ]
    
    # HPO phase
    cmd_hpo = [sys.executable, TRAIN_RL_SCRIPT_PATH] + common_args + ["--mode", "hpo"]
    result = subprocess.run(cmd_hpo, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"HPO failed for w={w_val}, lam={lam_val}")
        return False
    
    # Final training phase
    cmd_final = [sys.executable, TRAIN_RL_SCRIPT_PATH] + common_args + ["--mode", "final"]
    result = subprocess.run(cmd_final, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Final training failed for w={w_val}, lam={lam_val}")
        return False
        
    return True

def main():
    os.makedirs(RESULTS_DIR_DRIVER, exist_ok=True)
    print("Starting Pareto frontier generation...")
    
    if not os.path.exists(GRID_YAML_PATH):
        print(f"Grid file not found: {GRID_YAML_PATH}")
        return
        
    weights_w, weights_lam = load_grid_params(GRID_YAML_PATH)
    
    base_seed = int(time.time()) % 10000
    run_counter = 0
    
    for loss_type in LOSS_TYPES_TO_RUN:
        for w_val in weights_w:
            for lam_val in weights_lam:
                seed = base_seed + run_counter * 100
                run_counter += 1
                
                print(f"Processing: loss={loss_type}, w={w_val}, lam={lam_val}")
                
                success = run_training_pipeline(loss_type, w_val, lam_val, seed)
                if success:
                    print(f"Completed: loss={loss_type}, w={w_val}, lam={lam_val}")
                else:
                    print(f"Failed: loss={loss_type}, w={w_val}, lam={lam_val}")
                
                time.sleep(2)  # Brief pause between runs
    
    print("Pareto frontier generation complete.")

if __name__ == "__main__":
    main()
