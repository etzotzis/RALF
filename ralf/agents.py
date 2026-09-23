# -*- coding: utf-8 -*-
"""RALF agent classes: Forecaster, StockManager, Coordinator.

RALF pairs each agent with the RL algorithm that best matches the structure
of its decision problem, rather than using one algorithm for all three:

- Forecaster: on-policy Advantage Actor-Critic (A2C). Demand forecasting is a
  continuous, stochastic prediction task, so a Gaussian policy with a
  learnable standard deviation lets the agent express its own uncertainty,
  and advantage-based credit assignment keeps variance low.
- StockManager: off-policy Deep Deterministic Policy Gradient (DDPG).
  Ordering decisions have a deterministic optimum given the current state,
  so a replay buffer and target networks let the agent reuse past
  experience and train more stably than on-policy methods would.
- Coordinator: PPO with clipped trust-region updates. The Coordinator's
  meta-reward-weighting problem is inherently non-stationary (its targets
  shift as the operational agents improve), so PPO's clipped objective
  bounds how far each policy update can move.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ralf.buffer import ExperienceBuffer, ReplayBuffer


class Forecaster(nn.Module):
    """Forecaster Agent: on-policy Advantage Actor-Critic (A2C)."""

    def __init__(self, obs_dim=10, hidden_dim=64, lr=3e-4,
                 device='cpu', buffer_size=2000, max_demand=500, gamma=0.99):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.max_demand = max_demand

        # Policy network: outputs mean of Gaussian
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1], scale to [0, max_demand]
        ).to(self.device)

        # Value network: outputs scalar value estimate
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # Learnable log standard deviation for exploration
        self.log_std = nn.Parameter(
            torch.ones(1, device=self.device) * -0.5
        )

        self.policy_optimizer = optim.Adam(
            list(self.policy_net.parameters()) + [self.log_std],
            lr=lr
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=lr
        )

        self.buffer = ExperienceBuffer(buffer_size)

        self.discount_factor = gamma
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.02
        self.update_count = 0

    def select_action(self, observation):
        """Select action and store in buffer (with placeholder reward)"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean_normalized = self.policy_net(obs_tensor)
            mean = mean_normalized * self.max_demand

            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0, self.max_demand)

            log_prob = dist.log_prob(action)

            value = self.value_net(obs_tensor)

        self.buffer.add(
            obs=observation,
            action=float(action.item()),
            log_prob=float(log_prob.item()),
            reward=0.0,
            value=float(value.item())
        )

        return float(action.item())

    def store_transition(self, reward, next_observation=None, done=False):
        """Update reward for the last stored experience"""
        if len(self.buffer.rewards) > 0:
            rewards_list = list(self.buffer.rewards)
            rewards_list[-1] = reward
            self.buffer.rewards = deque(rewards_list, maxlen=self.buffer.max_size)

    def update(self, next_value=0.0, n_epochs=1):
        """
        A2C update: a single on-policy gradient step over the whole episode
        batch (no importance-sampling ratio, no clipping, no epoch reuse of
        stale data — the defining property that separates A2C from PPO).

        Called once at end of episode with all stored experiences.
        """
        if len(self.buffer) == 0:
            return 0.0, 0.0

        self.buffer.compute_returns(self.discount_factor, next_value)
        batch = self.buffer.get_all()

        obs_batch = torch.FloatTensor(batch['observations']).to(self.device)
        actions_batch = torch.FloatTensor(batch['actions']).unsqueeze(1).to(self.device)
        returns_batch = torch.FloatTensor(batch['returns']).to(self.device)

        mean_normalized = self.policy_net(obs_batch)
        mean = mean_normalized * self.max_demand
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        log_probs = dist.log_prob(actions_batch)

        values = self.value_net(obs_batch).squeeze(-1)

        advantages = returns_batch - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ==================== A2C POLICY LOSS ====================
        policy_loss = -(log_probs.squeeze(-1) * advantages).mean()
        # ===========================================================

        entropy = dist.entropy().mean()

        value_loss = torch.mean((values - returns_batch) ** 2)

        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy
        )

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_([self.log_std], 0.1)

        for name, param in self.policy_net.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}, zeroing it out")
                param.grad.zero_()

        self.policy_optimizer.step()
        self.value_optimizer.step()

        self.buffer.clear()
        self.update_count += 1

        return float(policy_loss.item()), float(value_loss.item())


class StockManager(nn.Module):
    """Stock Manager Agent: off-policy Deep Deterministic Policy Gradient (DDPG).

    The actor outputs a continuous order quantity directly (normalized to
    [0, 1], then scaled to [0, max_order_quantity] units). This matches
    RetailMARLExplainable.step()'s fallback branch, which treats any
    non-integer 'stock_manager' action as an already-computed unit count
    rather than a discrete action index.

    Vanilla single-critic DDPG at a shared learning rate reliably collapsed
    the actor toward near-zero order quantities in this environment's
    asymmetric, spiky reward landscape (empirically: the critic's Q-estimate
    became monotonically decreasing in the action, a well-known DDPG
    overestimation failure mode). Two things were needed to fix it:

    1. TD3's stabilization tricks: twin critics (the TD target uses the min
       of two independently-trained Q-networks, killing the single-critic
       overestimation bias), delayed actor updates (the actor and target
       networks update once every `policy_delay` critic steps) and target
       action smoothing (small clipped noise on the target action before
       computing the TD target, smoothing out sharp, exploitable Q-peaks).
    2. A much slower actor learning rate than critic learning rate
       (actor_lr << critic_lr): letting the critic settle into an accurate
       value estimate before the actor chases it, rather than both moving
       at the same pace. TD3's tricks alone were not sufficient here; this
       asymmetry was what actually stopped the collapse.
    """

    def __init__(self, obs_dim=11, hidden_dim=64, actor_lr=1e-5, critic_lr=3e-4,
                 device='cpu', buffer_size=10000, batch_size=32,
                 max_order_quantity=500, tau=0.001,
                 noise_std=0.1, noise_decay=0.995, noise_min=0.01, gamma=0.99,
                 policy_delay=2, target_noise_std=0.05, target_noise_clip=0.1):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.max_order_quantity = max_order_quantity
        self.batch_size = batch_size
        self.tau = tau
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip

        def make_actor():
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # Deterministic action in [0, 1], scaled to units on return
            ).to(self.device)

        def make_critic():
            return nn.Sequential(
                nn.Linear(obs_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ).to(self.device)

        self.actor = make_actor()
        self.actor_target = make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = make_critic()
        self.critic_2 = make_critic()
        self.critic_1_target = make_critic()
        self.critic_2_target = make_critic()
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr
        )

        self.buffer = ReplayBuffer(buffer_size)

        self.discount_factor = gamma
        self.update_count = 0
        self._critic_step_count = 0

        self._pending_obs = None
        self._pending_action_norm = None

    def select_action(self, observation):
        """Select a deterministic action plus decaying Gaussian exploration
        noise, and stash (obs, action) until store_transition supplies the
        reward, next observation and done flag needed to build the replay
        transition."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_norm = float(self.actor(obs_tensor).item())

        noisy_action_norm = float(np.clip(
            action_norm + np.random.normal(0, self.noise_std), 0.0, 1.0
        ))

        self._pending_obs = observation
        self._pending_action_norm = noisy_action_norm

        # Returned as a plain float (units, not an index) so the environment's
        # fallback branch treats it as the order quantity directly.
        return float(noisy_action_norm * self.max_order_quantity)

    def store_transition(self, reward, next_observation=None, done=False):
        """Finalize the pending transition and push it into the replay buffer"""
        if self._pending_obs is None:
            return

        next_obs = next_observation if next_observation is not None else self._pending_obs

        self.buffer.add(
            obs=self._pending_obs,
            action=self._pending_action_norm,
            reward=reward,
            next_obs=next_obs,
            done=float(done)
        )

        self._pending_obs = None
        self._pending_action_norm = None

    def _soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def update(self, next_value=0.0, n_epochs=4):
        """Off-policy DDPG update with TD3-style stabilization: sample
        minibatches from the replay buffer, update both critics via a
        twin-critic, smoothed TD target, and update the actor plus target
        networks only once every `policy_delay` critic steps."""
        if len(self.buffer) < self.batch_size:
            self._decay_noise()
            return 0.0, 0.0

        actor_losses = []
        critic_losses = []

        for _ in range(n_epochs):
            batch = self.buffer.sample(self.batch_size)

            obs = torch.FloatTensor(batch['observations']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
            next_obs = torch.FloatTensor(batch['next_observations']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

            # ==================== TWIN CRITIC UPDATE ====================
            with torch.no_grad():
                target_action = self.actor_target(next_obs)
                smoothing_noise = torch.clamp(
                    torch.randn_like(target_action) * self.target_noise_std,
                    -self.target_noise_clip, self.target_noise_clip
                )
                target_action = torch.clamp(target_action + smoothing_noise, 0.0, 1.0)

                target_q1 = self.critic_1_target(torch.cat([next_obs, target_action], dim=1))
                target_q2 = self.critic_2_target(torch.cat([next_obs, target_action], dim=1))
                target_q = torch.min(target_q1, target_q2)
                y = rewards + self.discount_factor * (1.0 - dones) * target_q

            q1 = self.critic_1(torch.cat([obs, actions], dim=1))
            q2 = self.critic_2(torch.cat([obs, actions], dim=1))
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 0.5
            )
            self.critic_optimizer.step()
            critic_losses.append(float(critic_loss.item()))
            # ================================================================

            self._critic_step_count += 1

            # ================ DELAYED ACTOR + TARGET UPDATE ================
            if self._critic_step_count % self.policy_delay == 0:
                actor_loss = -self.critic_1(torch.cat([obs, self.actor(obs)], dim=1)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self._soft_update(self.actor_target, self.actor)
                self._soft_update(self.critic_1_target, self.critic_1)
                self._soft_update(self.critic_2_target, self.critic_2)

                actor_losses.append(float(actor_loss.item()))
            # =================================================================

        self._decay_noise()
        self.update_count += 1

        mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
        return mean_actor_loss, float(np.mean(critic_losses))

    def _decay_noise(self):
        self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)


class Coordinator(nn.Module):
    """Coordinator Agent: PPO with clipped trust-region updates for stable
    meta-learning over the reward weights of the other two agents."""

    def __init__(self, obs_dim=12, hidden_dim=64, lr=1e-4,
                 device='cpu', buffer_size=2000,
                 batch_size=32, ppo_epsilon=0.2, gamma=0.99):
        super().__init__()
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ppo_epsilon = ppo_epsilon

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        ).to(self.device)

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        self.log_std = nn.Parameter(
            torch.ones(3, device=self.device) * -1.0
        )

        self.policy_optimizer = optim.Adam(
            list(self.policy_net.parameters()) + [self.log_std],
            lr=lr
        )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.buffer = ExperienceBuffer(buffer_size)

        self.discount_factor = gamma
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.05
        self.update_count = 0

    def select_action(self, observation):
        """Select action and store in buffer"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean = self.policy_net(obs_tensor)
            mean = torch.clamp(mean, -5, 5)
            if torch.isnan(mean).any():
                print("NaN detected in policy output, using safe default")
                mean = torch.where(torch.isnan(mean), torch.ones_like(mean), mean)
            std = torch.exp(self.log_std)
            std = torch.clamp(std, 1e-4, 1.0)
            try:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            except ValueError as e:
                print(f"Distribution error: {e}")
                action = torch.ones_like(mean) * 1.0
            action = torch.sigmoid(action) + 0.5
            action = torch.clamp(action, 0.5, 1.5)
            try:
                log_prob = dist.log_prob(action).sum(dim=-1)
            except Exception:
                log_prob = torch.zeros(1, device=self.device)
            value = self.value_net(obs_tensor)

        self.buffer.add(
            obs=observation,
            action=action.squeeze().detach().cpu().numpy(),
            log_prob=float(log_prob.item()),
            reward=0.0,
            value=float(value.item())
        )

        return action.squeeze().detach().cpu().numpy()

    def store_transition(self, reward, next_observation=None, done=False):
        """Update reward for the last stored experience"""
        if len(self.buffer.rewards) > 0:
            rewards_list = list(self.buffer.rewards)
            rewards_list[-1] = reward
            self.buffer.rewards = deque(rewards_list, maxlen=self.buffer.max_size)

    def update(self, next_value=0.0, n_epochs=4):
        """Update with PPO clipping + batch learning"""
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        self.buffer.compute_returns(self.discount_factor, next_value)

        policy_losses = []
        value_losses = []

        for epoch in range(n_epochs):
            batch = self.buffer.sample_batch(self.batch_size)

            obs_batch = torch.FloatTensor(batch['observations']).to(self.device)
            actions_batch = torch.FloatTensor(batch['actions']).to(self.device)
            old_log_probs_batch = torch.FloatTensor(batch['log_probs']).to(self.device)
            returns_batch = torch.FloatTensor(batch['returns']).to(self.device)

            mean = self.policy_net(obs_batch)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)

            values = self.value_net(obs_batch).squeeze()
            advantages = returns_batch - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)

            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            entropy = dist.entropy().sum(dim=-1).mean()
            value_loss = torch.mean((values - returns_batch) ** 2)

            total_loss = (
                policy_loss +
                self.value_loss_coef * value_loss -
                self.entropy_coef * entropy
            )

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_([self.log_std], 0.5)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))

        self.buffer.clear()
        self.update_count += 1

        return np.mean(policy_losses), np.mean(value_losses)
