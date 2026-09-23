# -*- coding: utf-8 -*-
"""Baseline policies for the paper's Table 12 (RALF vs. classical and
single-algorithm baselines).

Two families:

1. Classical inventory-control policies -- (s, S) and base-stock -- which
   don't learn. Since RetailMARLExplainable.step() adds ordered units to
   inventory the same day (no lead time), a plain reorder-point /
   order-up-to-level rule is enough to drive the environment and produce
   the same business metrics (profit, fill rate, overstock/stockout rate,
   inventory turnover) the learned agents are evaluated on.

   Default thresholds (s=150, S=300) are set from the dataset's own demand
   statistics: mean daily Units Sold is ~136 (std ~109), so s reflects
   roughly one day of average demand and S roughly two.

2. A homogeneous all-PPO multi-agent baseline (PPOForecaster,
   PPOStockManager), matching the thesis's own "MAPPO" comparison point:
   the same three-agent structure as RALF, but every agent uses PPO
   instead of RALF's heterogeneous A2C/DDPG/PPO split. The Coordinator is
   RALF's own PPO Coordinator, reused unchanged.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ralf.buffer import ExperienceBuffer


def run_classical_policy(df, policy, config, retail_config=None, forecast_value=400.0):
    """Drive the environment with a rule-based order policy (no training).

    forecast_value is fed as the Forecaster's action purely because
    RetailMARLExplainable.step() uses last_forecasted_demand to compute a
    hard inventory cap (max_reasonable_inventory); a generous constant
    keeps that cap from artificially constraining the classical policy.
    The Coordinator's weights don't affect env dynamics/profit (only the
    RL agents' own training reward), so a neutral [1, 1, 1] is used.
    """
    from ralf.config import RetailConfig
    from ralf.environment import RetailMARLExplainable
    from ralf.metrics import AdvancedMetricsCalculator

    if retail_config is None:
        retail_config = RetailConfig()

    env = RetailMARLExplainable(df, config=retail_config, seed=config.seed)
    metrics_calc = AdvancedMetricsCalculator()
    metrics_history = []

    for _ in range(config.num_episodes):
        obs, info = env.reset()
        terminated = False
        step_count = 0
        while not terminated and step_count < env.episode_length:
            order_units = policy.order_quantity(env.current_inventory)
            obs, rewards, terminated, _, info = env.step({
                'forecaster': np.array([forecast_value]),
                'stock_manager': order_units,
                'coordinator': np.array([1.0, 1.0, 1.0], dtype=np.float32),
            })
            step_count += 1
        metrics_history.append(metrics_calc.calculate_detailed_metrics(env, retail_config))

    return metrics_history


def train_homogeneous_ppo(df, config, retail_config=None):
    """Train the thesis's 'MAPPO' baseline: all three agents use PPO
    (Forecaster and Stock Manager here, Coordinator reused unchanged from
    ralf.agents since it was already PPO in the heterogeneous RALF too)."""
    from ralf.agents import Coordinator
    from ralf.config import RetailConfig
    from ralf.environment import RetailMARLExplainable
    from ralf.metrics import AdvancedMetricsCalculator

    if retail_config is None:
        retail_config = RetailConfig()

    env = RetailMARLExplainable(df, config=retail_config, seed=config.seed)

    forecaster = PPOForecaster(obs_dim=10, hidden_dim=config.hidden_dim,
                                lr=config.forecaster_lr, device=config.device,
                                max_demand=retail_config.max_demand, gamma=config.gamma,
                                ppo_epsilon=config.ppo_epsilon)
    stock_manager = PPOStockManager(obs_dim=11, hidden_dim=config.hidden_dim,
                                     lr=config.stock_manager_critic_lr, device=config.device,
                                     gamma=config.gamma, ppo_epsilon=config.ppo_epsilon)
    coordinator = Coordinator(obs_dim=12, hidden_dim=config.hidden_dim,
                               lr=config.coordinator_lr, device=config.device,
                               ppo_epsilon=config.ppo_epsilon, gamma=config.gamma)

    metrics_calc = AdvancedMetricsCalculator()
    metrics_history = []

    for _ in range(config.num_episodes):
        obs, info = env.reset()
        terminated = False
        step_count = 0
        while not terminated and step_count < env.episode_length:
            f_action = forecaster.select_action(obs['forecaster'])
            s_action = stock_manager.select_action(obs['stock_manager'])
            c_action = coordinator.select_action(obs['coordinator'])

            obs, rewards, terminated, _, info = env.step({
                'forecaster': np.array([f_action]),
                'stock_manager': s_action,
                'coordinator': c_action
            })

            forecaster.store_transition(rewards['forecaster'])
            stock_manager.store_transition(rewards['stock_manager'])
            coordinator.store_transition(rewards['coordinator'])
            step_count += 1

        forecaster.update(n_epochs=config.n_epochs)
        stock_manager.update(n_epochs=config.n_epochs)
        coordinator.update(n_epochs=config.n_epochs)

        metrics_history.append(metrics_calc.calculate_detailed_metrics(env, retail_config))

    return forecaster, stock_manager, coordinator, env, metrics_history


class SSPolicy:
    """(s, S) policy: if inventory < s, order up to S; otherwise order nothing."""

    def __init__(self, s=150.0, S=300.0, max_order_quantity=500):
        self.s = s
        self.S = S
        self.max_order_quantity = max_order_quantity

    def order_quantity(self, current_inventory):
        if current_inventory < self.s:
            return float(np.clip(self.S - current_inventory, 0, self.max_order_quantity))
        return 0.0


class BaseStockPolicy:
    """Base-stock policy: order up to a fixed level every period."""

    def __init__(self, base_stock_level=300.0, max_order_quantity=500):
        self.base_stock_level = base_stock_level
        self.max_order_quantity = max_order_quantity

    def order_quantity(self, current_inventory):
        return float(np.clip(self.base_stock_level - current_inventory, 0, self.max_order_quantity))


class PPOForecaster(nn.Module):
    """Homogeneous-baseline Forecaster: continuous Gaussian policy trained
    with PPO clipping (RALF's actual Forecaster uses A2C instead)."""

    def __init__(self, obs_dim=10, hidden_dim=64, lr=3e-4,
                 device='cpu', buffer_size=2000,
                 batch_size=32, ppo_epsilon=0.2, max_demand=500, gamma=0.99):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ppo_epsilon = ppo_epsilon
        self.max_demand = max_demand

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        self.log_std = nn.Parameter(torch.ones(1, device=self.device) * -0.5)

        self.policy_optimizer = optim.Adam(
            list(self.policy_net.parameters()) + [self.log_std], lr=lr
        )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.buffer = ExperienceBuffer(buffer_size)

        self.discount_factor = gamma
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.02
        self.update_count = 0

    def select_action(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean = self.policy_net(obs_tensor) * self.max_demand
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            action = torch.clamp(dist.sample(), 0, self.max_demand)
            log_prob = dist.log_prob(action)
            value = self.value_net(obs_tensor)

        self.buffer.add(obs=observation, action=float(action.item()),
                         log_prob=float(log_prob.item()), reward=0.0, value=float(value.item()))
        return float(action.item())

    def store_transition(self, reward, next_observation=None, done=False):
        if len(self.buffer.rewards) > 0:
            rewards_list = list(self.buffer.rewards)
            rewards_list[-1] = reward
            self.buffer.rewards = deque(rewards_list, maxlen=self.buffer.max_size)

    def update(self, next_value=0.0, n_epochs=4):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        self.buffer.compute_returns(self.discount_factor, next_value)
        policy_losses, value_losses = [], []

        for _ in range(n_epochs):
            batch = self.buffer.sample_batch(self.batch_size)
            obs_batch = torch.FloatTensor(batch['observations']).to(self.device)
            actions_batch = torch.FloatTensor(batch['actions']).unsqueeze(1).to(self.device)
            old_log_probs_batch = torch.FloatTensor(batch['log_probs']).unsqueeze(1).to(self.device)
            returns_batch = torch.FloatTensor(batch['returns']).to(self.device)

            mean = self.policy_net(obs_batch) * self.max_demand
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_batch)

            values = self.value_net(obs_batch).squeeze()
            advantages = returns_batch - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
            policy_loss = -torch.min(ratio * advantages.unsqueeze(1),
                                      clipped_ratio * advantages.unsqueeze(1)).mean()

            entropy = dist.entropy().mean()
            value_loss = torch.mean((values - returns_batch) ** 2)
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_([self.log_std], 0.1)
            self.policy_optimizer.step()
            self.value_optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))

        self.buffer.clear()
        self.update_count += 1
        return float(np.mean(policy_losses)), float(np.mean(value_losses))


class PPOStockManager(nn.Module):
    """Homogeneous-baseline Stock Manager: discrete Categorical policy
    trained with PPO clipping (RALF's actual Stock Manager uses DDPG
    instead). Action space matches the environment's discretization: 51
    actions, action_i -> action_i * (max_order_quantity / 50) units."""

    def __init__(self, obs_dim=11, hidden_dim=64, lr=3e-4,
                 device='cpu', buffer_size=2000,
                 batch_size=32, ppo_epsilon=0.2, num_actions=51, gamma=0.99):
        super().__init__()
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ppo_epsilon = ppo_epsilon
        self.num_actions = num_actions

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        ).to(self.device)

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.buffer = ExperienceBuffer(buffer_size)

        self.discount_factor = gamma
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.2
        self.update_count = 0

    def select_action(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(obs_tensor)
            value = self.value_net(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        self.buffer.add(obs=observation, action=int(action.item()),
                         log_prob=float(log_prob.item()), reward=0.0, value=float(value.item()))
        return int(action.item())

    def store_transition(self, reward, next_observation=None, done=False):
        if len(self.buffer.rewards) > 0:
            rewards_list = list(self.buffer.rewards)
            rewards_list[-1] = reward
            self.buffer.rewards = deque(rewards_list, maxlen=self.buffer.max_size)

    def update(self, next_value=0.0, n_epochs=4):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        self.buffer.compute_returns(self.discount_factor, next_value)
        policy_losses, value_losses = [], []

        for _ in range(n_epochs):
            batch = self.buffer.sample_batch(self.batch_size)
            obs_batch = torch.FloatTensor(batch['observations']).to(self.device)
            actions_batch = torch.LongTensor(batch['actions']).to(self.device)
            old_log_probs_batch = torch.FloatTensor(batch['log_probs']).to(self.device)
            returns_batch = torch.FloatTensor(batch['returns']).to(self.device)

            logits = self.policy_net(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze()

            values = self.value_net(obs_batch).squeeze()
            advantages = returns_batch - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            entropy = -(probs * log_probs).sum(dim=-1).mean()
            value_loss = torch.mean((values - returns_batch) ** 2)
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))

        self.buffer.clear()
        self.update_count += 1
        return float(np.mean(policy_losses)), float(np.mean(value_losses))
