# -*- coding: utf-8 -*-
"""Configuration dataclasses for the RALF environment and training experiments."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RetailConfig:
    """Environment configuration"""
    purchase_price_multiplier: float = 4.0  # Buy at 1/4 price
    holding_cost_per_unit: float = 0.5      # Daily holding cost
    stockout_penalty_multiplier: float = 5.0  # 3x revenue lost
    overstock_penalty_multiplier: float = 0.2  # 0.1 per unit per day
    forecast_error_weight: float = 0.3      # Forecaster loss weight

    episode_length: int = 60  # Days per episode
    max_order_quantity: int = 500  # Max units per order
    max_demand: int = 500  # Max demand estimate

    # Update frequency
    forecaster_update_freq: int = 1
    stock_manager_update_freq: int = 1
    coordinator_update_freq: int = 1


@dataclass
class ExperimentConfig:
    """Training experiment configuration"""
    num_episodes: int = 10000
    num_seeds: int = 1
    seed: int = 42
    device: str = 'cpu'
    visualize_every: int = 100
    checkpoint_every: int = 200

    # Agent hyperparameters
    forecaster_lr: float = 1e-4
    # Stock Manager (DDPG/TD3) needs asymmetric actor/critic rates: a fast
    # critic and a much slower actor. A shared, single-rate DDPG update
    # reliably collapsed the actor to near-zero order quantities in this
    # environment's asymmetric, spiky reward landscape (see agents.py).
    stock_manager_critic_lr: float = 3e-4
    stock_manager_actor_lr: float = 1e-5
    coordinator_lr: float = 1e-4
    hidden_dim: int = 64
    gamma: float = 0.99  # Discount factor, shared across all three agents

    # PPO Clipping
    ppo_epsilon: float = 0.2          # Clipping range: [1-eps, 1+eps]

    # Replay buffer
    buffer_size: int = 2000
    batch_size: int = 32
    n_epochs: int = 4

    experiment_name: str = "ralf_base"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
