# -*- coding: utf-8 -*-
"""Episode metrics calculators."""

from typing import Any, Dict

import numpy as np

from ralf.config import RetailConfig


class MetricsCalculator:
    """Calculate comprehensive metrics from episode history"""

    @staticmethod
    def calculate_episode_metrics(env, config: RetailConfig) -> Dict[str, float]:
        """Calculate all metrics for one episode"""

        if not env.episode_history:
            return {}

        history = env.episode_history

        total_profit = env.cumulative_profit
        daily_avg_profit = total_profit / max(1, len(history))

        mae = env.cumulative_mae / max(1, len(history))

        mape_values = []
        rmse_values = []
        for h in history:
            actual = h['info'].get('actual_demand', 1)
            forecasted = h['info'].get('forecasted_demand', 0)
            if actual > 0:
                mape_values.append(abs(forecasted - actual) / actual)
            rmse_values.append((forecasted - actual) ** 2)

        mape = (np.mean(mape_values) * 100) if mape_values else 0
        rmse = np.sqrt(np.mean(rmse_values)) if rmse_values else 0
        forecast_accuracy = max(0, (1 - mape / 100) * 100) if mape > 0 else 100

        stockout_rate = (env.stockout_count / max(1, len(history))) * 100
        overstock_rate = (env.overstock_count / max(1, len(history))) * 100

        avg_inventory = np.mean([h['info'].get('inventory_after', 0) for h in history])

        total_units_sold = sum([h['info'].get('units_sold', 0) for h in history])
        inventory_turnover = total_units_sold / max(1, avg_inventory * len(history))

        total_demand = sum([h['info'].get('actual_demand', 0) for h in history])
        service_level = (total_units_sold / total_demand * 100) if total_demand > 0 else 0

        return {
            'total_profit': total_profit,
            'daily_avg_profit': daily_avg_profit,
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'forecast_accuracy': forecast_accuracy,
            'stockout_rate': stockout_rate,
            'overstock_rate': overstock_rate,
            'avg_inventory': avg_inventory,
            'inventory_turnover': inventory_turnover,
            'service_level': service_level,
            'total_stockouts': env.stockout_count,
            'total_overstock_days': env.overstock_count
        }


class AdvancedMetricsCalculator:
    """Calculate comprehensive metrics from episode history with detailed tracking"""

    @staticmethod
    def calculate_detailed_metrics(env, config: RetailConfig) -> Dict[str, Any]:
        """Calculate all metrics for one episode with financial breakdown"""

        if not env.episode_history:
            return {}

        history = env.episode_history

        # ===== FINANCIAL METRICS =====
        revenues = []
        profits = []
        costs = []
        holding_costs = []

        for h in history:
            info = h['info']
            units_sold = info.get('units_sold', 0)
            price = info.get('actual_demand', 50)  # Using as price proxy
            revenue = units_sold * price * 0.95  # Approximate
            revenues.append(revenue)

            daily_profit = info.get('daily_profit', 0)
            profits.append(daily_profit)

            ordered = h['actions']['stock_manager'] * 10
            cost = ordered * (price / 4)
            costs.append(cost)

            inventory = info.get('inventory_after', 0)
            h_cost = inventory * 0.5
            holding_costs.append(h_cost)

        total_revenue = sum(revenues)
        total_profit = env.cumulative_profit
        daily_avg_profit = total_profit / max(1, len(history))

        # ===== DEMAND FORECASTING METRICS =====
        mae = env.cumulative_mae / max(1, len(history))

        mape_values = []
        rmse_values = []
        for h in history:
            actual = h['info'].get('actual_demand', 1)
            forecasted = h['info'].get('forecasted_demand', 0)
            if actual > 0:
                mape_values.append(abs(forecasted - actual) / actual)
            rmse_values.append((forecasted - actual) ** 2)

        mape = (np.mean(mape_values) * 100) if mape_values else 0
        rmse = np.sqrt(np.mean(rmse_values)) if rmse_values else 0
        forecast_accuracy = max(0, (1 - mape / 100) * 100) if mape > 0 else 100

        # ===== INVENTORY METRICS =====
        stockout_rate = (env.stockout_count / max(1, len(history))) * 100
        overstock_rate = (env.overstock_count / max(1, len(history))) * 100

        total_stockouts = sum([h['info'].get('stockout_units', 0) for h in history])
        total_overstocks = sum([h['info'].get('overstock_units', 0) for h in history])

        avg_inventory = np.mean([h['info'].get('inventory_after', 0) for h in history])

        total_units_sold = sum([h['info'].get('units_sold', 0) for h in history])
        inventory_turnover = total_units_sold / max(1, avg_inventory * len(history))

        total_demand = sum([h['info'].get('actual_demand', 0) for h in history])
        service_level = (total_units_sold / total_demand * 100) if total_demand > 0 else 0

        # ===== REWARD METRICS =====
        forecaster_rewards = [h['rewards'].get('forecaster', 0) for h in history]
        stock_mgr_rewards = [h['rewards'].get('stock_manager', 0) for h in history]
        coord_rewards = [h['rewards'].get('coordinator', 0) for h in history]

        avg_forecaster_reward = np.mean(forecaster_rewards) if forecaster_rewards else 0
        avg_stock_mgr_reward = np.mean(stock_mgr_rewards) if stock_mgr_rewards else 0
        avg_coord_reward = np.mean(coord_rewards) if coord_rewards else 0
        avg_total_reward = (avg_forecaster_reward + avg_stock_mgr_reward + avg_coord_reward) / 3

        return {
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'daily_avg_profit': daily_avg_profit,
            'total_cost': sum(costs),
            'total_holding_cost': sum(holding_costs),

            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'forecast_accuracy': forecast_accuracy,

            'stockout_rate': stockout_rate,
            'overstock_rate': overstock_rate,
            'total_stockouts': total_stockouts,
            'total_overstocks': total_overstocks,
            'avg_inventory': avg_inventory,
            'inventory_turnover': inventory_turnover,
            'service_level': service_level,
            'total_stockout_days': env.stockout_count,
            'total_overstock_days': env.overstock_count,

            'avg_forecaster_reward': avg_forecaster_reward,
            'avg_stock_mgr_reward': avg_stock_mgr_reward,
            'avg_coordinator_reward': avg_coord_reward,
            'avg_total_reward': avg_total_reward
        }
