# RALF — Retail Adaptive Learning Framework

Multi-Agent Reinforcement Learning (MARL) system for retail inventory management. Three heterogeneous agents — a **Forecaster** (A2C), a **Stock Manager** (DDPG), and a **Coordinator** (PPO) — learn to jointly forecast demand, place replenishment orders, and balance profit against stockout/overstock risk on a day-by-day retail simulation. Each agent uses the RL algorithm best suited to its decision problem rather than a single shared algorithm.

This project is a restructured, script-runnable port of a Google Colab notebook (`ralf_v3_5.py`), with the Colab-specific bits (`!pip install`, hardcoded Google Drive paths) removed and the code split into a proper package.

## Project layout

```
RALF/
├── main.py                 # CLI entrypoint: load data, train, plot, analyze
├── requirements.txt
├── data/                   # Put your retail_store_inventory.csv here
├── outputs/                # Generated plots, metrics CSVs, snapshots
└── ralf/
    ├── config.py            # RetailConfig, ExperimentConfig dataclasses
    ├── data.py              # CSV loading helper
    ├── environment.py       # RetailMARLExplainable (gymnasium.Env)
    ├── buffer.py            # ExperienceBuffer (on-policy) + ReplayBuffer (off-policy)
    ├── agents.py            # Forecaster (A2C), StockManager (DDPG), Coordinator (PPO)
    ├── networks.py          # Shared PolicyNetwork / ValueNetwork blocks
    ├── metrics.py            # MetricsCalculator, AdvancedMetricsCalculator
    ├── visualization.py     # TrainingVisualizer (matplotlib dashboards)
    ├── training.py          # train_heterogeneous_marl_with_timing()
    └── analysis.py          # Agent decision snapshots + algorithm comparison table
```

## Setup

```bash
cd ~/Downloads/RALF
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Expected columns (spaces/slashes are auto-standardized):

```
Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing,
Seasonality, Category, Region, Inventory Level, Units Sold, Demand Forecast
```

The dataset is the Kaggle ["Retail Store Inventory Forecasting Dataset"](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset).

If `data/retail_store_inventory.csv` (or the path passed via `--data`) doesn't exist, `main.py`
automatically downloads it from Kaggle using `kagglehub`. This requires a Kaggle API token — see
[Kaggle auth setup](#kaggle-auth-setup) below. Pass `--no-auto-download` to disable this and require
a local file instead.

To place the file manually instead, put your CSV at `data/retail_store_inventory.csv` (or pass `--data <path>`).

### Kaggle auth setup

`kagglehub` needs Kaggle API credentials. Get a token from https://www.kaggle.com/settings →
"Create New Token" (downloads `kaggle.json`), then either:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

or set environment variables instead:

```bash
export KAGGLE_USERNAME=<your-username>
export KAGGLE_KEY=<your-api-key>
```

## Running

```bash
python main.py --episodes 1000
```

This downloads the dataset automatically on first run (cached by kagglehub afterward) and trains
for 1000 episodes. To use a local file instead, pass `--data data/retail_store_inventory.csv`.

Useful flags:

- `--episodes N` — number of training episodes (default 1000)
- `--seed N` — random seed (default 42)
- `--device cpu|cuda` — defaults to CUDA if available
- `--output-dir DIR` — where plots/CSVs/snapshots are written (default `outputs/`)
- `--skip-plots` — skip generating the matplotlib dashboards
- `--skip-snapshots` — skip the post-training agent decision snapshots
- `--no-auto-download` — don't fetch the dataset from Kaggle; require it to already exist at `--data`

## What it produces

In `outputs/`:

- `00_comprehensive_dashboard.png` ... `05_training_time.png` — training curves (profit, revenue, stockouts, overstocks, forecast error, rewards, timing)
- `ralf_training_metrics.csv` — per-episode metrics
- `ralf_test_snapshots.csv` — mid-episode agent decisions on fresh test episodes
- `ralf_algorithm_comparison.csv` — RALF vs. MAPPO/MADDPG/MAA2C/MA-POCA comparison table (only the RALF row is measured; the rest are illustrative multipliers for context, not real runs of those algorithms)

## Notes on the port

- Fixed a bug in the original notebook's `reset()` (referenced an undefined `idx` instead of the episode's first-day index).
- Added `price`/`discount` to the environment's `info` dict so the agent-snapshot report shows real values instead of defaults.
- Removed duplicate/inline exploratory Colab cells (ad-hoc data inspection printouts); the same logic now lives in `ralf/data.py`.
