# Driver script for executing RL-based hedging strategies.

import subprocess
import yaml
import os
import pandas as pd
import logging
import sys
import time
from tqdm import tqdm
from utils import execute_command, load_completed_runs, check_required_files, load_grid


_project_root_driver = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRID_YAML_PATH = os.path.join(_project_root_driver, "agents/grid.yaml")
TRAIN_RL_SCRIPT_PATH = os.path.join(_project_root_driver, "agents", "train_ppo.py") 
RESULTS_DIR_DRIVER = os.path.join(_project_root_driver, "results")
PARETO_RAW_CSV_DRIVER = os.path.join(RESULTS_DIR_DRIVER, "pareto_raw.csv")
DRIVER_LOG_FILE = os.path.join(RESULTS_DIR_DRIVER, "driver.log")

LOSS_TYPES_TO_RUN = ["mse", "abs"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] DRIVER: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DRIVER_LOG_FILE, mode='a')
    ]
)

# Retrieves completed evaluations from a CSV file.
def get_completed_evals(csv_path):
    return load_completed_runs(csv_path, "eval_done")

def run_command(command_list):
    return execute_command(command_list, "SUBPROCESS")


def main():
    os.makedirs(RESULTS_DIR_DRIVER, exist_ok=True)
    logging.info("Starting Pareto Frontier Generation Driver.")
    
    if not check_required_files([GRID_YAML_PATH, TRAIN_RL_SCRIPT_PATH]):
        return
        

    weights_pnl, weights_lam = load_grid(GRID_YAML_PATH)
    if not weights_pnl or not weights_lam:
        logging.error("Grid weights for 'w' or 'lam' are empty in grid.yaml.")
        return

    completed_eval_triplets = get_completed_evals(PARETO_RAW_CSV_DRIVER)
    logging.info(f"Found {len(completed_eval_triplets)} already evaluated (loss_type, w, lam) triplets in {PARETO_RAW_CSV_DRIVER}.")

    driver_base_seed = int(time.time()) % 10000 
    
    run_counter = 0
    for loss_type_val in tqdm(LOSS_TYPES_TO_RUN, desc="Loss Types", ncols=100, position=0):
        for i_w, w_val_in in enumerate(tqdm(weights_pnl, desc=f"w-grid (loss={loss_type_val})", ncols=100, position=1, leave=False)):
            for i_l, lam_val_in in enumerate(tqdm(weights_lam, desc=f"lam-grid (loss={loss_type_val}, w={w_val_in})", ncols=100, position=2, leave=False)):
                w_val = float(w_val_in)
                lam_val = float(lam_val_in)
                
                pair_base_seed = driver_base_seed + run_counter * 100 
                run_counter +=1

                logging.info(f"Processing: loss={loss_type_val}, w={w_val}, lam={lam_val} with base pair_seed={pair_base_seed}")
                
                current_triplet_tuple = (loss_type_val, w_val, lam_val)
                if current_triplet_tuple in completed_eval_triplets:
                    logging.info(f"Skipping {current_triplet_tuple} as it's already logged as 'eval_done'.")
                    continue

                common_args = ["--loss_type", loss_type_val, "--w", str(w_val), "--lam", str(lam_val), "--seed", str(pair_base_seed)]

                logging.info(f"Starting HPO for {current_triplet_tuple}")
                cmd_hpo = [sys.executable, TRAIN_RL_SCRIPT_PATH] + common_args + ["--mode", "hpo"]
                if not run_command(cmd_hpo):
                    logging.error(f"HPO failed for {current_triplet_tuple}. Skipping to next triplet.")
                    continue
                logging.info(f"HPO completed for {current_triplet_tuple}")

                logging.info(f"Starting Final Training for {current_triplet_tuple}")
                cmd_final = [sys.executable, TRAIN_RL_SCRIPT_PATH] + common_args + ["--mode", "final"]
                if not run_command(cmd_final):
                    logging.error(f"Final Training failed for {current_triplet_tuple}. Skipping to next triplet.")
                    continue
                logging.info(f"Final Training completed for {current_triplet_tuple}")
                
                logging.info(f"Starting Evaluation for {current_triplet_tuple}")
                cmd_eval = [sys.executable, TRAIN_RL_SCRIPT_PATH] + common_args + ["--mode", "eval"]
                if not run_command(cmd_eval):
                    logging.error(f"Evaluation failed for {current_triplet_tuple}. Results for this triplet might be incomplete.")
                else:
                    logging.info(f"Evaluation completed and results logged for {current_triplet_tuple}")
                
                logging.info(f"Finished processing for {current_triplet_tuple}")
                time.sleep(5) 

    logging.info("All loss_type, w, lam combinations processed. Pareto Frontier Generation Driver finished.")

if __name__ == "__main__":
    main()
