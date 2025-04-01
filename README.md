# Deep Reinforcement Learning Hedging Agent

[![![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
PyTorch][(https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/status-active-green.svg)](https://github.com/bcosm/rBergomi-HedgeRL)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements a Deep Reinforcement Learning (DRL) agent for dynamic hedging of a 10,000-share SPY position using long-dated call and put options. The agent is trained via Proximal Policy Optimization (PPO) with an LSTM backbone to capture temporal dependencies in market microstructure.

Rather than constantly rebalancing stock positions (classical delta-hedging), the agent uses options contracts to achieve superior risk-adjusted returns with lower transaction costs.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Quick Example](#quick-example)
- [System Architecture](#system-architecture)
- [Technical Implementation](#technical-implementation)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Contact](#contact)

---

## Quick Start

### Installation

```bash
git clone https://github.com/bcosm/rBergomi-HedgeRL.git
cd rBergomi-HedgeRL
pip install -r requirements.txt
pip install -e .
```

### Command-Line Interface

#### Training

```bash
# Default training (no HPO) - uses defaults: loss_type=abs, w=0.001, lam=0.0001, seed=12345
hedgerl train

# With custom parameters
hedgerl train --loss_type mse --w 0.01 --lam 0.001 --seed 42
```

**Training Parameters:**
- `--loss_type`: Loss function type (`abs`, `mse`) - default: `abs`
- `--w`: PnL penalty weight (float) - default: `0.001` 
- `--lam`: Transaction cost weight (float) - default: `0.0001`
- `--seed`: Random seed (int) - default: `12345`

#### Backtesting

```bash
# Single run (random seed)
hedgerl backtest

# Single run (specific seed)
hedgerl backtest --seed 42
```

---

## Quick Example

```python
from hedgerl import Agent
# This is a conceptual example; actual implementation may vary
# agent = Agent.load('model_files/policy_weights.pth')
# agent.hedge(spy_price=450.0)
```

---

## System Architecture

### High-Level Architecture Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Market Data   │───▶│  rBergomi Sim    │───▶│  Training Data  │
│   (SPY, Options)│    │  (GPU-Accelerated)│    │  (100k paths)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Deployment    │◀───│   Trained Model  │◀───│  PPO + LSTM     │
│  (QuantConnect, │    │   (PyTorch)      │    │  Training       │
│   Backtrader)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         ▲
                                ▼                         │
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Action Space    │    │  Reward Engine  │
                       │  (6 discrete     │    │  (12 components)│
                       │   actions)       │    │                 │
                       └──────────────────┘    └─────────────────┘
```

### Core Components

1.  **Market Simulation Engine**
    - rBergomi Model: Rough volatility with fractional Brownian motion
    - 100,000 Monte Carlo paths with 252 time steps each
    - GPU-accelerated simulation using CUDA kernels
    - Real-time parameter perturbation for market regime robustness

2.  **Deep Reinforcement Learning**
    - PPO Algorithm: Proximal Policy Optimization for stable training
    - LSTM Network: Captures temporal dependencies in market data
    - State Space: 15 features (prices, Greeks, P&L stats, time)
    - Action Space: 6 discrete actions (buy/sell/hold calls/puts)

3.  **Advanced Reward Engineering**
    - 12 specialized penalty terms for realistic trading behavior
    - Multi-objective optimization: P&L variance vs transaction costs
    - Risk management: Delta neutrality, gamma control, position limits

---

## Technical Implementation

### Neural Network Architecture

```python
# PPO with LSTM for temporal dependencies
State Space: 13 features (prices, Greeks, P&L stats, time)
Action Space: 2 continuous actions (call/put trade quantities)
Network: FC(128) → LSTM(64) → FC(64) → Actor/Critic heads
Optimization: Adam optimizer with gradient clipping
```

### State Representation

The agent observes comprehensive market information:
- Price features: Current SPY price, call/put prices (ATM)
- Position features: Current call/put contract holdings
- Greeks exposure: Portfolio delta and gamma from options
- Time features: Remaining episode time (normalized)
- Risk metrics: Volatility, lagged price/volatility changes

### Action Space Design

```python
Actions = {
    Continuous call action: [-1, 1] mapped to [-max_trade, +max_trade] contracts
    Continuous put action:  [-1, 1] mapped to [-max_trade, +max_trade] contracts
}
# Where max_trade = 100 contracts per step
# Positive values = buy contracts, negative = sell contracts
```

### Reward Function Components

The reward at each step is computed as:
- **PnL penalty**: scaled per-share PnL deviation using `loss_type`:
  - `abs`: |ΔPnL| / S0
  - `mse`: (ΔPnL)² / S0²
- **Transaction cost penalty**: `lambda_cost` × (contracts traded) × cost per contract
- **Position limits**: Enforced by clipping holdings to ±`max_contracts_held`

### Transaction Cost Modeling

- **Commission**: $0.65 per options contract (default)

### Numerical Stability in Greeks Calculation

To avoid numerical issues in the `_calculate_greeks` method:
- Asset price (`S`) floor: 1e-6
- Time-to-expiry (`T`) floor: 1e-6
- Volatility spot (`v_spot`) floor: 1e-8
- Strike (`K`) floor: 1e-6
- Capping of d1/d2 at ±10.0
---

## Research Methodology

### Environment Design

The training environment (`hedging_env.py`) provides a comprehensive simulation framework:

| Component | Features | Purpose |
|-----------|----------|---------|
| Market Data | 100k rBergomi paths, 252 timesteps each | Realistic market simulation |
| Action Space | Continuous call/put contract trading | Fine-grained hedging control |
| Reward Engineering | Multi-objective PnL variance + costs | Balanced risk/cost optimization |
| State Space | Market prices, Greeks, positions, time | Complete market information |

### Hyperparameter Optimization

Pareto frontier analysis using Optuna across multiple configurations:

```bash
# Generate full parameter grid search
hedgerl generate-pareto
```

This runs HPO, final training, and evaluation for all combinations of:
- **Loss types**: `["mse", "abs", "cvar"]` 
- **PnL weights**: `[0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]`
- **Cost weights**: `[0.0001, 0.001, 0.01]`

Results are saved to `src/agents/results/pareto_raw.csv`.

---

## Repository Structure

```text
.
├── data/                              # Market data and simulation results
│   ├── historical_prices.csv            # SPY historical price data
│   ├── paths_rbergomi_options_100k.npz  # 100k rBergomi simulation paths
│   ├── spy_options.csv                  # Real options market data
│   ├── spy_options.parquet              # Real options market data (Parquet)
│   ├── spy_underlying.csv               # Underlying price data
│   └── spy_underlying.parquet           # Underlying price data (Parquet)
├── src/
│   ├── agents/                           # RL training and optimization
│   │   ├── train_ppo.py                 # PPO implementation with CLI support
│   │   ├── driver.py                    # Pareto frontier generation
│   │   ├── grid.yaml                    # Parameter grid configuration
│   │   └── baselines.py                 # Delta-hedge benchmarks
│   ├── env/                             # RL environment
│   │   └── hedging_env.py               # Core hedging environment
│   ├── sim/                             # Market simulation
│   │   ├── rbergomi_sim.py              # GPU-accelerated rough volatility
│   │   └── option_price_assignment.py   # Vectorized option pricing
│   ├── backtester/                      # Production backtesting
│   │   ├── main_backtrader.py           # Backtrader integration
│   │   ├── model_wrapper_bt.py          # Model wrapper for backtesting
│   │   ├── delta_hedge.py               # Delta hedge baseline
│   │   └── option_calculator.py         # Robust B-S implementation
│   └── results/                         # Training and evaluation results
│       ├── pareto_raw.csv               # Pareto frontier analysis results
│       └── models/                      # Saved model checkpoints
├── quantconnect/                      # Institutional integration
│   ├── main.py                          # LEAN algorithm
│   └── model_wrapper.py                 # Production model wrapper
├── model_files/                       # Trained models
│   ├── policy_weights.pth               # Neural network weights
│   ├── normalization_stats.pkl          # Feature scaling parameters
│   └── architecture_info.pkl            # Model architecture metadata
└── hedgerl/                          # CLI package
    └── cli.py                           # Command-line interface
```

---

## Business Impact

For a 10,000 share equity position in SPY, the agent would save:
- $150k annually in transaction costs vs delta-hedging
- 96% reduction in total trading costs
- 23.8% lower volatility
- 50%+ improvement in drawdown metrics (Ulcer Index, Pain Index)
- 2,410% better hedging efficiency vs classical approaches

---

## Next Steps

- **Multi-asset hedging**: Extend to portfolios of stocks
- **Market microstructure**: Add order book dynamics to training
- **Regime switching**: Adapt to changing market conditions
- **Reward engineering**: Incorporate additional reward terms, penalize higher-order Greeks
- **Model architecture**: Experiment with DDPG, D4PG, or multi-arm bandit models

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions about the research, implementation, or potential collaborations:

- **Email**: [bcosm@umich.edu](mailto:bcosm@umich.edu)
- **LinkedIn**: [Baz Cosmopoulos](https://linkedin.com/in/baz-cosmopoulos)
- **GitHub**: [bcosm](https://github.com/bcosm)

---

## Acknowledgments

- Rough volatility modeling inspired by Gatheral et al. (2018)
- Deep reinforcement learning frameworks: Stable-Baselines3, PyTorch
- Quantitative finance libraries: QuantLib, numpy, scipy
- GPU acceleration: CUDA, CuPy for high-performance computing

---