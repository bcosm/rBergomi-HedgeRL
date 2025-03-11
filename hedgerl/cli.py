#!/usr/bin/env python3
"""
Command Line Interface for HedgeRL Backtesting
"""

import argparse
import sys
import os
import random
from pathlib import Path


def find_project_root():
    """Find the project root directory by looking for model_files folder"""
    current = Path.cwd()
    
    # Check if we're already in the project directory
    if (current / "model_files").exists():
        return current
    
    # Look for the project directory structure
    for parent in [current] + list(current.parents):
        if (parent / "model_files").exists() and (parent / "src" / "backtester").exists():
            return parent
    
    # Last resort: check if we're in a subdirectory
    for item in current.rglob("model_files"):
        if item.is_dir() and (item.parent / "src" / "backtester").exists():
            return item.parent
    
    raise FileNotFoundError(
        "Could not find project root. Please run this command from the CantorRL directory "
        "or ensure the model_files and src/backtester directories exist."
    )


def run_backtest(num_seeds):
    """Run the backtest in loop mode with multiple seeds"""
    try:
        project_root = find_project_root()
        print(f"Found project root: {project_root}")
        
        # Verify required directories exist
        required_dirs = ["data", "model_files", "src/backtester"]
        missing_dirs = []
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"Error: Missing required directories: {', '.join(missing_dirs)}")
            print("Please ensure you are running from the CantorRL project directory.")
            return 1
        
        # Verify model files exist
        model_files = ["policy_weights.pth", "normalization_stats.pkl", "architecture_info.pkl"]
        missing_models = []
        for model_file in model_files:
            if not (project_root / "model_files" / model_file).exists():
                missing_models.append(model_file)
        
        if missing_models:
            print(f"Warning: Missing model files: {', '.join(missing_models)}")
            print("The backtest may not work correctly without these files.")
        
        # Verify data files exist
        data_files = ["spy_options.csv", "spy_underlying.csv"]
        missing_data = []
        for data_file in data_files:
            if not (project_root / "data" / data_file).exists():
                missing_data.append(data_file)
        
        if missing_data:
            print(f"Error: Missing data files: {', '.join(missing_data)}")
            print("Please ensure the required data files are in the data/ directory.")
            return 1
        
        # Change to project directory
        os.chdir(project_root)
        
        # Add src/backtester to path
        backtester_path = project_root / "src" / "backtester"
        if str(backtester_path) not in sys.path:
            sys.path.insert(0, str(backtester_path))
        
        # Import the backtesting module
        try:
            from main_backtrader import loop_runs
            import main_backtrader
        except ImportError as e:
            print(f"Error importing backtesting modules: {e}")
            print("Please ensure all required packages are installed:")
            print("  pip install -e .")
            return 1
        
        print(f"Running backtest loop with {num_seeds} seeds")
        print("This may take several minutes...")
        
        # Modify global variables for loop mode
        original_use_loop = main_backtrader.USE_LOOP
        original_num_seeds = main_backtrader.NUM_SEEDS
        
        main_backtrader.USE_LOOP = True
        main_backtrader.NUM_SEEDS = num_seeds
        
        try:
            loop_runs()
        finally:
            # Restore original values
            main_backtrader.USE_LOOP = original_use_loop
            main_backtrader.NUM_SEEDS = original_num_seeds
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1



def run_backtest_with_seed(seed):
    """Run the backtest with a specific seed"""
    try:
        project_root = find_project_root()
        print(f"Found project root: {project_root}")
        
        # Verify required directories exist
        required_dirs = ["data", "model_files", "src/backtester"]
        missing_dirs = []
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"Error: Missing required directories: {', '.join(missing_dirs)}")
            print("Please ensure you are running from the CantorRL project directory.")
            return 1
        
        # Verify model files exist
        model_files = ["policy_weights.pth", "normalization_stats.pkl", "architecture_info.pkl"]
        missing_models = []
        for model_file in model_files:
            if not (project_root / "model_files" / model_file).exists():
                missing_models.append(model_file)
        
        if missing_models:
            print(f"Warning: Missing model files: {', '.join(missing_models)}")
            print("The backtest may not work correctly without these files.")
        
        # Verify data files exist
        data_files = ["spy_options.csv", "spy_underlying.csv"]
        missing_data = []
        for data_file in data_files:
            if not (project_root / "data" / data_file).exists():
                missing_data.append(data_file)
        
        if missing_data:
            print(f"Error: Missing data files: {', '.join(missing_data)}")
            print("Please ensure the required data files are in the data/ directory.")
            return 1
        
        # Change to project directory
        os.chdir(project_root)
        
        # Add src/backtester to path
        backtester_path = project_root / "src" / "backtester"
        if str(backtester_path) not in sys.path:
            sys.path.insert(0, str(backtester_path))
        
        # Import the backtesting module
        try:
            from main_backtrader import one_run
            from delta_hedge import run as run_delta_hedge
            import json
            import pandas as pd
        except ImportError as e:
            print(f"Error importing backtesting modules: {e}")
            print("Please ensure all required packages are installed:")
            print("  pip install -e .")
            return 1
        
        print(f"Running single backtest with seed: {seed}")
        print("This may take a few minutes...")
        result = one_run(seed)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_training(args):
    """Run the training with or without HPO"""
    hpo_trials = args.hpo
    loss_type = args.loss_type
    w = args.w
    lam = args.lam
    seed = args.seed
    try:
        project_root = find_project_root()
        print(f"Found project root: {project_root}")
        
        # Verify required directories exist
        required_dirs = ["data", "src/agents"]
        missing_dirs = []
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"Error: Missing required directories: {', '.join(missing_dirs)}")
            print("Please ensure you are running from the CantorRL project directory.")
            return 1
        
        # Verify data files exist
        data_files = ["paths_rbergomi_options_100k.npz"]
        missing_data = []
        for data_file in data_files:
            if not (project_root / "data" / data_file).exists():
                missing_data.append(data_file)
        
        if missing_data:
            print(f"Error: Missing data files: {', '.join(missing_data)}")
            print("Please ensure the required data files are in the data/ directory.")
            return 1
        
        # Change to project directory
        os.chdir(project_root)
        
        # Add src/agents to path
        agents_path = project_root / "src" / "agents"
        if str(agents_path) not in sys.path:
            sys.path.insert(0, str(agents_path))
        
        # Import the training module
        try:
            import train_ppo as train_ppo_v3
        except ImportError as e:
            print(f"Error importing training modules: {e}")
            print("Please ensure all required packages are installed:")
            print("  pip install -e .")
            return 1
        
        if hpo_trials is None:
            # Final training mode (no HPO)
            print("Running final training (no HPO)...")
            print(f"Using parameters: loss_type={loss_type}, w={w}, lam={lam}, seed={seed}")
            print("This will use default hyperparameters or load best parameters from previous HPO run.")
            print("This may take a significant amount of time...")
            
            # Modify the MODE to 'final' temporarily
            original_mode = train_ppo_v3.MODE
            train_ppo_v3.MODE = "final"
            
            try:
                train_ppo_v3.main_cli(loss_type=loss_type, w=w, lam=lam, seed=seed)
                print("\nTraining completed successfully!")
                print("Model files have been saved to model_files/")
            finally:
                # Restore original mode
                train_ppo_v3.MODE = original_mode
        else:
            # HPO mode
            print(f"Running HPO training with {hpo_trials} trials...")
            print(f"Using parameters: loss_type={loss_type}, w={w}, lam={lam}, seed={seed}")
            print("This will optimize hyperparameters and then run final training.")
            print("This may take a very long time...")
            
            # Modify the constants temporarily
            original_mode = train_ppo_v3.MODE
            original_trials = train_ppo_v3.N_OPTUNA_TRIALS
            
            train_ppo_v3.MODE = "hpo"
            train_ppo_v3.N_OPTUNA_TRIALS = hpo_trials
            
            try:
                # Run HPO
                train_ppo_v3.main_cli(loss_type=loss_type, w=w, lam=lam, seed=seed)
                
                # Now run final training with best parameters
                print(f"\nHPO completed with {hpo_trials} trials!")
                print("Starting final training with optimized hyperparameters...")
                
                train_ppo_v3.MODE = "final"
                train_ppo_v3.main_cli(loss_type=loss_type, w=w, lam=lam, seed=seed)
                
                print("\nHPO + Final training completed successfully!")
                print("Model files have been saved to model_files/")
                print("HPO database saved to src/agents/results/optuna_dbs/")
            finally:
                # Restore original values
                train_ppo_v3.MODE = original_mode
                train_ppo_v3.N_OPTUNA_TRIALS = original_trials
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_pareto_generation():
    """Run the Pareto frontier generation"""
    try:
        project_root = find_project_root()
        print(f"Found project root: {project_root}")
        
        # Verify required directories exist
        required_dirs = ["data", "src/agents"]
        missing_dirs = []
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"Error: Missing required directories: {', '.join(missing_dirs)}")
            print("Please ensure you are running from the CantorRL project directory.")
            return 1
        
        # Verify data files exist
        data_files = ["paths_rbergomi_options_100k.npz"]
        missing_data = []
        for data_file in data_files:
            if not (project_root / "data" / data_file).exists():
                missing_data.append(data_file)
        
        if missing_data:
            print(f"Error: Missing data files: {', '.join(missing_data)}")
            print("Please ensure the required data files are in the data/ directory.")
            return 1
        
        # Verify grid.yaml exists
        grid_file = project_root / "src" / "agents" / "grid.yaml"
        if not grid_file.exists():
            print(f"Error: Grid configuration file not found: {grid_file}")
            print("Please ensure grid.yaml exists in src/agents/ directory.")
            return 1
        
        # Change to project directory
        os.chdir(project_root)
        
        # Add src/agents to path
        agents_path = project_root / "src" / "agents"
        if str(agents_path) not in sys.path:
            sys.path.insert(0, str(agents_path))
        
        # Import and run the driver
        try:
            import driver
        except ImportError as e:
            print(f"Error importing driver module: {e}")
            print("Please ensure all required packages are installed:")
            print("  pip install -e .")
            return 1
        
        print("Starting Pareto frontier generation...")
        print("This will run HPO, final training, and evaluation for multiple parameter combinations.")
        print("This process can take a very long time (hours to days depending on the grid size).")
        print("")
        
        # Run the driver
        driver.main()
        
        print("\nPareto frontier generation completed!")
        print("Results have been saved to src/agents/results/pareto_raw.csv")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred during Pareto generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HedgeRL Backtesting and Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hedgerl backtest                    # Single run with random seed vs delta hedge
  hedgerl backtest --loop 10         # Loop mode with 10 different seeds vs delta hedge
  hedgerl backtest --seed 123        # Single run with specific seed vs delta hedge
  hedgerl train                       # Train model without HPO (default params)
  hedgerl train --hpo                 # Train model with 10 HPO trials (default)
  hedgerl train --hpo 20             # Train model with 20 HPO trials
  hedgerl train --w 0.01 --lam 0.001 # Train with custom parameters
  hedgerl train --hpo 15 --loss_type cvar  # HPO with custom loss type
  hedgerl generate-pareto             # Generate Pareto frontier using parameter grid search
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", 
        help="Run the RL hedging backtest"
    )
    backtest_parser.add_argument(
        "--loop",
        type=int,
        help="Number of seeds to run in loop mode"
    )
    backtest_parser.add_argument(
        "--seed",
        type=int,
        help="Specific seed to use for single run"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train the RL hedging model"
    )
    train_parser.add_argument(
        "--hpo",
        type=int,
        nargs="?",
        const=10,  # Default value when --hpo is used without argument
        help="Run hyperparameter optimization with specified number of trials (default: 10)"
    )
    train_parser.add_argument(
        "--loss_type",
        type=str,
        choices=["mse", "abs", "cvar"],
        default="abs",
        help="Loss type for PnL penalty (default: abs)"
    )
    train_parser.add_argument(
        "--w",
        type=float,
        default=0.05,
        help="PnL penalty weight (default: 0.001)"
    )
    train_parser.add_argument(
        "--lam",
        type=float,
        default=0.001,
        help="Transaction cost weight (default: 0.0001)"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345)"
    )
    
    # Generate Pareto command
    pareto_parser = subparsers.add_parser(
        "generate-pareto",
        help="Generate Pareto frontier by running parameter grid search"
    )
    
    # Handle case where user just runs "hedgerl" without arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    args = parser.parse_args()
    
    if args.command == "backtest":
        if args.loop is not None:
            return run_backtest(args.loop)
        elif args.seed is not None:
            return run_backtest_with_seed(args.seed)
        else:
            # Default to single run with random seed
            seed = random.randint(1, 10000)
            return run_backtest_with_seed(seed)
    elif args.command == "train":
        return run_training(args)
    elif args.command == "generate-pareto":
        return run_pareto_generation()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
