# -*- coding: utf-8 -*-
"""Experience buffer for batch/PPO-style learning."""

from collections import deque

import numpy as np


class ExperienceBuffer:
    """
    Store experiences for batch learning and PPO training.

    Replaces immediate-update strategy with buffered batch learning.
    """

    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.log_probs = deque(maxlen=max_size)  # For PPO clipping
        self.rewards = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.returns = deque(maxlen=max_size)

    def add(self, obs, action, log_prob, reward, value):
        """Add experience to buffer"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def compute_returns(self, gamma=0.99, next_value=0.0):
        """
        Compute discounted returns (rewards-to-go)

        Needed for PPO: old_log_probs * (return - value_estimate)
        """
        returns = []
        cumulative_return = next_value

        for reward in reversed(list(self.rewards)):
            cumulative_return = reward + gamma * cumulative_return
            returns.insert(0, cumulative_return)

        self.returns = deque(returns, maxlen=self.max_size)

    def sample_batch(self, batch_size=32):
        """
        Sample random batch for training

        Returns dict with torch-ready arrays
        """
        if len(self) < batch_size:
            batch_size = len(self)

        indices = np.random.choice(len(self), batch_size, replace=False)

        return {
            'observations': np.array([self.observations[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'log_probs': np.array([self.log_probs[i] for i in indices]),
            'returns': np.array([self.returns[i] for i in indices]),
        }

    def get_all(self):
        """Return all stored experiences as arrays, preserving temporal order.

        Used by on-policy algorithms like A2C that train on the full episode
        batch in one pass rather than sampling random minibatches.
        """
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'returns': np.array(self.returns),
        }

    def __len__(self):
        return len(self.observations)

    def clear(self):
        """Clear all buffers for next episode"""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.returns.clear()


class ReplayBuffer:
    """Off-policy replay buffer for DDPG-style transition sampling.

    Unlike ExperienceBuffer (which stores a single episode's on-policy
    trajectory and is cleared after each update), ReplayBuffer persists
    transitions across episodes and is sampled uniformly at random.
    """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, action, reward, next_obs, done):
        """Add a full (s, a, r, s', done) transition to the buffer"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        """Sample a random minibatch of transitions"""
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.buffer[i] for i in indices))

        return {
            'observations': np.array(obs),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_observations': np.array(next_obs),
            'dones': np.array(dones),
        }

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
