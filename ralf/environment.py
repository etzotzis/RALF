# -*- coding: utf-8 -*-
"""Multi-Agent Retail Environment with standardized column handling."""

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from ralf.config import RetailConfig


class RetailMARLExplainable(gym.Env):
    """Multi-Agent Retail Environment with Fixed Column Handling"""

    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, config: RetailConfig = None, seed: int = 42):
        self.config = config or RetailConfig()

        self.df = df.copy()
        self.seed_val = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Standardize column names: spaces -> underscores, slashes -> underscores
        # 'Weather Condition' -> 'Weather_Condition'
        # 'Holiday/Promotion' -> 'Holiday_Promotion'
        # 'Competitor Pricing' -> 'Competitor_Pricing'
        # 'Inventory Level' -> 'Inventory_Level'
        # 'Units Sold' -> 'Units_Sold'
        # 'Demand Forecast' -> 'Demand_Forecast'
        self.df.columns = self.df.columns.str.replace(' ', '_').str.replace('/', '_')

        # Data preparation
        self._prepare_data()

        # Agents
        self.agents = ['forecaster', 'stock_manager', 'coordinator']

        # Episode state
        self.current_day = 0
        self.episode_length = self.config.episode_length
        self.episode_data_indices = []

        # Track forecaster's prediction for stock manager
        self.last_forecasted_demand = 0.0

        # Metrics
        self.cumulative_profit = 0.0
        self.cumulative_mae = 0.0
        self.forecast_count = 0
        self.stockout_count = 0
        self.overstock_count = 0

        # Episode history
        self.episode_history = []

        num_stock_actions = self.config.max_order_quantity // 10 + 1
        self.stock_action_step = self.config.max_order_quantity / (num_stock_actions - 1)

        print(f"Stock Manager: {num_stock_actions} discrete actions (0-{num_stock_actions-1})")
        print(f"  Mapping: action_i -> {self.stock_action_step:.2f} * i units")

        # Dynamic state tracking: represents CURRENT state that agents affect
        self.current_inventory = 0.0  # Running inventory (updates each step)
        self.current_day_price = 0.0  # Today's selling price
        self.current_day_demand = 0.0  # Today's actual demand

        # Define spaces
        self._define_spaces()

        print("Environment initialized successfully!")

    def _prepare_data(self):
        """Prepare and normalize dataset"""

        print("Preparing data with column standardization...")

        # Categorical encoding (using standardized names)
        if 'Weather_Condition' in self.df.columns:
            self.df['Weather_Condition'] = pd.factorize(self.df['Weather_Condition'])[0]
            print("  Encoded: Weather_Condition")

        if 'Holiday_Promotion' in self.df.columns:
            self.df['Holiday_Promotion'] = self.df['Holiday_Promotion'].astype(int)
            print("  Encoded: Holiday_Promotion")

        if 'Seasonality' in self.df.columns:
            self.df['Seasonality'] = pd.factorize(self.df['Seasonality'])[0]
            print("  Encoded: Seasonality")

        if 'Category' in self.df.columns:
            self.df['Category'] = pd.factorize(self.df['Category'])[0]
            print("  Encoded: Category")

        if 'Region' in self.df.columns:
            self.df['Region'] = pd.factorize(self.df['Region'])[0]
            print("  Encoded: Region")

        # Normalize numeric columns (using corrected column names)
        numeric_cols = [
            'Price', 'Discount', 'Competitor_Pricing', 'Demand_Forecast',
            'Inventory_Level', 'Units_Sold'
        ]

        print("\n  Normalizing numeric columns:")
        for col in numeric_cols:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[f'{col}_norm'] = (self.df[col] - min_val) / (max_val - min_val)
                    print(f"    {col}: [{min_val:.2f}, {max_val:.2f}] -> normalized")
                else:
                    self.df[f'{col}_norm'] = 0.5
                    print(f"    {col}: constant value, set to 0.5")
            else:
                # If column missing, create dummy normalized column
                self.df[f'{col}_norm'] = 0.5
                print(f"    {col}: NOT FOUND, created default 0.5")

        self.data_length = len(self.df)
        print(f"\nData prepared! Rows: {self.data_length}, Columns: {len(self.df.columns)}")
        print(f"  Available columns: {list(self.df.columns)}")

    def _define_spaces(self):
        """Define observation and action spaces"""
        self.observation_spaces = {
            'forecaster': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            'stock_manager': spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32),
            'coordinator': spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        }

        self.action_spaces = {
            'forecaster': spaces.Box(low=0, high=self.config.max_demand, shape=(1,), dtype=np.float32),
            'stock_manager': spaces.Discrete(self.config.max_order_quantity // 10 + 1),
            'coordinator': spaces.Box(low=0.5, high=1.5, shape=(3,), dtype=np.float32)
        }

        self.observation_space = self.observation_spaces
        self.action_space = self.action_spaces

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)

        self.current_day = 0
        self.cumulative_profit = 0.0
        self.cumulative_mae = 0.0
        self.forecast_count = 0
        self.stockout_count = 0
        self.overstock_count = 0
        self.last_forecasted_demand = 0.0
        self.episode_history = []

        # Random episode start
        max_start_idx = max(0, self.data_length - self.episode_length)
        start_idx = np.random.randint(0, max_start_idx + 1)
        self.episode_data_indices = list(range(start_idx, start_idx + self.episode_length))

        # Initialize state for this episode from the FIRST day of the episode
        first_day_idx = self.episode_data_indices[0]
        row = self.df.iloc[first_day_idx]

        def get_val(col, default=0.0):
            if col in row.index:
                val = row[col]
                return float(val) if not np.isnan(val) else default
            return default

        self.current_inventory = get_val('Inventory_Level', 50.0)
        self.current_day_price = get_val('Price', 100.0)
        self.current_day_demand = get_val('Demand_Forecast', 50.0)

        # Meta reward weights
        self.meta_reward_weights = {
            'profit': 1.0,
            'stockout_penalty': 1.0,
            'forecast_error': 1.0
        }

        observations = self._get_observations()
        return observations, {}

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations for all agents"""
        if self.current_day >= len(self.episode_data_indices):
            return {
                'forecaster': np.zeros(10, dtype=np.float32),
                'stock_manager': np.zeros(11, dtype=np.float32),
                'coordinator': np.zeros(12, dtype=np.float32)
            }

        idx = self.episode_data_indices[self.current_day]
        row = self.df.iloc[idx]

        def get_val(col, default=0.0):
            """Safely get value from row, return default if missing"""
            if col in row.index:
                val = row[col]
                return float(val) if not np.isnan(val) else default
            return default

        # ===== Calculate historical demand features =====
        historical_demands = []
        historical_window = min(7, self.current_day)

        for i in range(max(0, self.current_day - historical_window), self.current_day):
            if i < len(self.episode_data_indices):
                hist_idx = self.episode_data_indices[i]
                hist_row = self.df.iloc[hist_idx]
                hist_demand = hist_row['Demand_Forecast'] if 'Demand_Forecast' in hist_row.index else 50.0
                historical_demands.append(float(hist_demand))

        if len(historical_demands) > 0:
            demand_7day_avg = np.mean(historical_demands)
            if len(historical_demands) > 1:
                demand_trend = (historical_demands[-1] - historical_demands[0]) / max(1.0, historical_demands[0])
            else:
                demand_trend = 0.0
            demand_std = np.std(historical_demands)
        else:
            demand_7day_avg = 50.0
            demand_trend = 0.0
            demand_std = 10.0

        demand_7day_avg_norm = np.clip(demand_7day_avg / self.config.max_demand, 0, 1)
        demand_trend_norm = np.clip((demand_trend + 1) / 2, 0, 1)
        demand_volatility_norm = np.clip(demand_std / (self.config.max_demand / 2), 0, 1)

        # ========== FORECASTER: Market conditions + Demand history ==========
        forecaster_obs = np.array([
            get_val('Price_norm', 0.5),
            get_val('Discount', 10) / 100.0,
            get_val('Weather_Condition', 0.0) / 10.0,
            get_val('Holiday_Promotion', 0.0),
            get_val('Competitor_Pricing_norm', 0.5),
            get_val('Seasonality', 0.0) / 10.0,
            demand_7day_avg_norm,
            demand_trend_norm,
            demand_volatility_norm,
            get_val('Category', 0.0) / 10.0,
        ], dtype=np.float32)

        # ========== STOCK MANAGER: Inventory + Forecast ==========
        # Gets forecaster's prediction from LAST step
        last_forecast_normalized = self.last_forecasted_demand / max(1, self.config.max_demand)

        inventory_level_norm = get_val('Inventory_Level_norm', 0.5)

        inventory_level = self.current_inventory
        forecasted_demand = max(1, self.last_forecasted_demand)

        # "Overstocking multiple": if inventory >= forecast*2.5, we're WAY too full
        if inventory_level >= forecasted_demand * 2.5:
            inventory_excess_signal = 1.0  # HIGH signal = don't order
        else:
            inventory_excess_signal = max(0, (inventory_level - forecasted_demand) / (forecasted_demand + 1.0))
            inventory_excess_signal = np.clip(inventory_excess_signal, 0, 1)

        stock_manager_obs = np.array([
            inventory_level_norm,
            inventory_excess_signal,
            last_forecast_normalized,
            get_val('Price_norm', 0.5),
            get_val('Discount', 10) / 100.0,
            get_val('Competitor_Pricing_norm', 0.5),
            get_val('Units_Sold_norm', 0.5),
            get_val('Holiday_Promotion', 0.0),
            get_val('Weather_Condition', 0.0) / 10.0,
            get_val('Category', 0.0) / 10.0,
            get_val('Region', 0.0) / 10.0,
        ], dtype=np.float32)

        # ========== COORDINATOR: System state + performance ==========
        avg_stockout_rate = self.stockout_count / max(1, self.forecast_count)
        avg_overstock_rate = self.overstock_count / max(1, self.forecast_count)
        avg_mae = self.cumulative_mae / max(1, self.forecast_count)

        coordinator_obs = np.array([
            np.clip(self.cumulative_profit / 10000.0, -1, 1),
            get_val('Inventory_Level_norm', 0.5),
            last_forecast_normalized,
            np.clip(avg_mae / 100.0, 0, 1),
            np.clip(avg_stockout_rate, 0, 1),
            np.clip(avg_overstock_rate, 0, 1),
            get_val('Price_norm', 0.5),
            get_val('Units_Sold_norm', 0.5),
            get_val('Category', 0.0) / 10.0,
            get_val('Region', 0.0) / 10.0,
            get_val('Seasonality', 0.0) / 10.0,
            get_val('Holiday_Promotion', 0.0)
        ], dtype=np.float32)

        return {
            'forecaster': forecaster_obs,
            'stock_manager': stock_manager_obs,
            'coordinator': coordinator_obs
        }

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, bool, Dict]:
        """Execute one step in environment"""
        terminated = self.current_day >= self.episode_length
        truncated = False

        if terminated:
            observations = self._get_observations()
            return observations, {'forecaster': 0., 'stock_manager': 0., 'coordinator': 0.}, terminated, truncated, {}

        idx = self.episode_data_indices[self.current_day]
        row = self.df.iloc[idx]

        # Parse actions
        forecasted_demand = float(actions['forecaster'][0]) if isinstance(actions['forecaster'], np.ndarray) else float(actions['forecaster'])
        forecasted_demand = np.clip(forecasted_demand, 0, self.config.max_demand)

        # Scale the discrete action to actual units (0 -> 0, max_index -> max_order_quantity)
        stock_action_index = actions['stock_manager']
        if isinstance(stock_action_index, (int, np.integer)):
            ordered_units = int(stock_action_index * self.stock_action_step)
        else:
            ordered_units = int(stock_action_index)  # Fallback

        ordered_units = int(np.clip(ordered_units, 0, self.config.max_order_quantity))

        # Store forecasted demand for NEXT step
        self.last_forecasted_demand = forecasted_demand

        coord_action = actions.get('coordinator', np.array([1.0, 1.0, 1.0]))
        self.meta_reward_weights['profit'] = float(coord_action[0])
        self.meta_reward_weights['stockout_penalty'] = float(coord_action[1])
        self.meta_reward_weights['forecast_error'] = float(coord_action[2])

        def get_val(col, default=0.0):
            if col in row.index:
                val = row[col]
                return float(val) if not np.isnan(val) else default
            return default

        # Use TRUE demand from dataset (Demand_Forecast column is ground truth)
        # Use RUNNING state from previous step (or initialization)
        current_inventory = self.current_inventory
        price = self.current_day_price
        true_demand = get_val('Demand_Forecast', 50.0)
        discount = get_val('Discount', 0.1)

        # Update inventory and sales
        new_inventory = current_inventory + ordered_units
        units_sold = min(true_demand, new_inventory)

        # Hard cap on inventory (prevents runaway overstocking)
        max_reasonable_inventory = self.last_forecasted_demand * 4.0

        if new_inventory > max_reasonable_inventory:
            excess = new_inventory - max_reasonable_inventory
            ordered_units = max(0, ordered_units - excess)
            new_inventory = current_inventory + ordered_units

            if excess > 0:
                print(f"Day {self.current_day}: Rejected {excess:.1f} units (inventory would exceed {max_reasonable_inventory:.0f})")

        # Additional hard inventory cap (prevents runaway overstocking)
        max_allowed_inventory = self.config.max_demand * 1.5  # 1.5x demand is max reasonable
        if new_inventory > max_allowed_inventory:
            ordered_units = max(0, max_allowed_inventory - current_inventory)
            new_inventory = current_inventory + ordered_units
        units_sold = max(0, units_sold)
        ending_inventory = new_inventory - units_sold

        # Update state for NEXT step
        self.current_inventory = ending_inventory

        # Financial calculations
        discount_decimal = discount / 100.0
        revenue = units_sold * price * (1 - discount_decimal)
        purchase_cost = ordered_units * (price / self.config.purchase_price_multiplier)
        holding_cost = ending_inventory * self.config.holding_cost_per_unit
        daily_profit = revenue - purchase_cost - holding_cost

        # Penalties
        stockout_units = max(0, true_demand - units_sold)
        overstock_units = max(0, ending_inventory - true_demand)

        stockout_penalty = stockout_units * self.config.stockout_penalty_multiplier
        overstock_penalty = overstock_units * self.config.overstock_penalty_multiplier

        forecast_error = abs(forecasted_demand - true_demand)

        # Update metrics
        self.cumulative_profit += daily_profit
        self.cumulative_mae += forecast_error
        self.forecast_count += 1

        if stockout_units > 0:
            self.stockout_count += 1
        if overstock_units > 0:
            self.overstock_count += 1

        # ===== 1. FORECASTER REWARD: Accuracy + Impact on Business =====
        max_possible_error = self.config.max_demand  # 500
        accuracy_ratio = 1.0 - (forecast_error / max_possible_error)
        base_reward = accuracy_ratio * 100.0

        if forecast_error <= true_demand * 0.1:
            accuracy_bonus = 50.0
        elif forecast_error <= true_demand * 0.2:
            accuracy_bonus = 25.0
        else:
            accuracy_bonus = 0.0

        # Penalize forecaster for causing stockouts/overstocks
        forecast_impact_penalty = 0.0
        if forecasted_demand < true_demand and stockout_units > 0:
            forecast_impact_penalty = min(forecast_error * 0.5, 50.0)
        elif forecasted_demand > true_demand and overstock_units > 0:
            forecast_impact_penalty = min(forecast_error * 0.3, 30.0)

        forecaster_reward = ((base_reward + accuracy_bonus - forecast_impact_penalty)
                            * self.meta_reward_weights['forecast_error'])

        # ===== 2. STOCK MANAGER REWARD =====
        normalized_profit = daily_profit / 100.0

        days_of_inventory = current_inventory / max(1, self.last_forecasted_demand)

        if days_of_inventory >= 3.0 and ordered_units > 0:
            penalty_for_excessive_order = ordered_units * (days_of_inventory - 2.0) * 3.0
        else:
            penalty_for_excessive_order = 0.0

        # stockout_penalty_multiplier defaults to 5.0, scaled by /10 so the
        # default reproduces the previously-hardcoded 0.5 factor exactly.
        stockout_cost = stockout_units * price * (self.config.stockout_penalty_multiplier / 10.0)
        severe_stockout_penalty = stockout_cost / 100.0
        excess_penalty = 0

        overstock_holding_cost = overstock_units * self.config.overstock_penalty_multiplier
        enhanced_overstock_penalty = overstock_holding_cost * 2.0

        stock_manager_reward = (
            normalized_profit * 0.8
            - (severe_stockout_penalty * self.meta_reward_weights['stockout_penalty'] * 3.0)
            - enhanced_overstock_penalty * 1.5
            - excess_penalty
        ) * self.meta_reward_weights['profit']

        # ===== 3. COORDINATOR REWARD: System-Wide Performance =====
        avg_profit_per_day = self.cumulative_profit / max(1, self.forecast_count)
        normalized_avg_profit = avg_profit_per_day / 100.0

        stockout_rate_pct = (self.stockout_count / max(1, self.forecast_count)) * 100
        if stockout_rate_pct > 10:
            stockout_penalty_score = -(stockout_rate_pct ** 1.5)
        else:
            stockout_penalty_score = -stockout_rate_pct

        overstock_rate_pct = (self.overstock_count / max(1, self.forecast_count)) * 100
        overstock_penalty_score = -overstock_rate_pct * 0.5

        service_level = (units_sold / max(1, true_demand)) * 100
        if service_level >= 95:
            service_bonus = 20.0
        elif service_level >= 90:
            service_bonus = 10.0
        else:
            service_bonus = 0.0

        coordinator_reward = (
            normalized_avg_profit * 2.0
            + stockout_penalty_score * self.meta_reward_weights['stockout_penalty']
            + overstock_penalty_score
            + service_bonus
        )

        rewards = {
            'forecaster': float(forecaster_reward),
            'stock_manager': float(stock_manager_reward),
            'coordinator': float(coordinator_reward)
        }

        info = {
            'day': self.current_day,
            'actual_demand': true_demand,
            'forecasted_demand': forecasted_demand,
            'ordered_units': ordered_units,
            'inventory_before': current_inventory,
            'inventory_after': ending_inventory,
            'units_sold': units_sold,
            'daily_profit': daily_profit,
            'cumulative_profit': self.cumulative_profit,
            'cumulative_mae': self.cumulative_mae,
            'stockout_count': self.stockout_count,
            'overstock_count': self.overstock_count,
            'forecast_error': forecast_error,
            'stockout_units': stockout_units,
            'overstock_units': overstock_units,
            'price': price,
            'discount': discount,
        }

        self.episode_history.append({
            'day': self.current_day,
            'actions': actions.copy(),
            'rewards': rewards.copy(),
            'info': info.copy()
        })

        self.current_day += 1

        observations = self._get_observations()
        return observations, rewards, terminated, truncated, info
