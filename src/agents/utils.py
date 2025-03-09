import subprocess
import logging
import pandas as pd
import os
import yaml


def load_grid(yaml_path):
    """Load grid configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        grid_config = yaml.safe_load(f)
    return grid_config.get('w', []), grid_config.get('lam', [])


def execute_command(command_list, log_prefix="SUBPROCESS"):
    """Execute a subprocess command with proper logging and error handling."""
    logging.info(f"Executing: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            logging.info(f"  {log_prefix}: {line.decode().strip()}")
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            logging.error(f"Command failed with exit code {return_code}: {' '.join(command_list)}")
            return False
    except Exception as e:
        logging.error(f"Exception during command execution: {e}")
        return False
    return True


def load_completed_runs(csv_path, status_filter="eval_done"):
    """Load completed runs from CSV to avoid re-processing."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
        
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return completed
            
        if 'loss_type' in df.columns and 'w' in df.columns and 'lam' in df.columns and 'status' in df.columns:
            for _, row in df.iterrows():
                if row['status'] == status_filter:
                    completed.add((str(row['loss_type']), float(row['w']), float(row['lam'])))
    except pd.errors.EmptyDataError:
        logging.info(f"{csv_path} is empty.")
    except Exception as e:
        logging.error(f"Error reading {csv_path}: {e}")
    
    return completed


def load_completed_baseline_runs(csv_path, target_algo_name, status_filter="eval_done"):
    """Load completed baseline runs for a specific algorithm."""
    completed_wl_pairs = set()
    if not os.path.exists(csv_path):
        return completed_wl_pairs
        
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return completed_wl_pairs
            
        if 'algo' in df.columns and 'w' in df.columns and 'lam' in df.columns and 'status' in df.columns:
            for _, row in df.iterrows():
                if row['algo'] == target_algo_name and row['status'] == status_filter:
                    completed_wl_pairs.add((float(row['w']), float(row['lam'])))
    except pd.errors.EmptyDataError:
        logging.info(f"Baseline CSV {csv_path} for {target_algo_name} is empty.")
    except Exception as e:
        logging.error(f"Error reading baseline CSV {csv_path} for {target_algo_name}: {e}")
    
    return completed_wl_pairs


def check_required_files(file_paths):
    """Check if all required files exist, log errors for missing ones."""
    missing_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            logging.error(f"Required file not found: {file_path}")
    
    return len(missing_files) == 0
