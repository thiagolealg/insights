
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime
from strategy_analyzer import StrategyAnalyzer

DATA_FILE = "win_full_data.parquet"
STRATEGIES_FILE = "saved_strategies.json"
TARGET_SHARPE = 2.3

class StrategyOptimizer:
    def __init__(self, strategy_type):
        self.strategy_type = strategy_type
        print(f"Loading data... Strategy: {strategy_type}")
        self.analyzer = StrategyAnalyzer(DATA_FILE, strategy_type=strategy_type)
        self.analyzer.run_backtest()
        
    def get_saved_strategies(self):
        if os.path.exists(STRATEGIES_FILE):
            try:
                with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_strategy(self, strategy_config):
        strategies = self.get_saved_strategies()
        strategies.append(strategy_config)
        with open(STRATEGIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
        print(f"Saved optimized strategy: {strategy_config['name']}")

    def optimize(self, inverted=False):
        mode_label = "Inverted" if inverted else "Normal"
        print(f"\n--- Optimizing {self.strategy_type} ({mode_label}) ---")
        
        # Reset filters
        self.analyzer.reset_filters()
        
        # Set base filters
        current_filters = {
            'direction': 'all',
            'invert': inverted,
            'hours': list(range(9, 18)), # 9 to 17
            'weekdays': list(range(5))   # Mon to Fri
        }
        
        # Initial Check
        self.analyzer.apply_filters(current_filters)
        summary = self.analyzer.get_summary(use_filtered=True)
        print(f"Initial State: Sharpe={summary.get('sharpe_ratio', 0):.2f}, Profit={summary.get('total_profit_points', 0):.2f}, Trades={summary.get('total_trades', 0)}")
        
        if summary.get('sharpe_ratio', 0) >= TARGET_SHARPE:
            print("Initial strategy already meets target!")
            self._save_result(current_filters, summary, mode_label)
            return

        best_sharpe = summary.get('sharpe_ratio', 0)
        best_filters = current_filters.copy()
        best_summary = summary

        while True:
            # 1. Analyze Breakdown
            stats_h = self.analyzer.get_stats_by_hour(use_filtered=True)
            stats_w = self.analyzer.get_stats_by_weekday(use_filtered=True)
            
            worst_metric = -float('inf')
            worst_type = None # 'hour' or 'weekday'
            worst_value = None
            
            # Find worst Hour
            for item in stats_h:
                if item['trades'] == 0: continue
                profit = item['total_profit']
                sharpe = item['sharpe_ratio']
                
                # Metric: Negative profit is prioritized. Then low Sharpe.
                if profit < 0:
                    metric = abs(profit) * 1000 # Heavily weight neg profit
                    if metric > worst_metric:
                        worst_metric = metric
                        worst_type = 'hour'
                        worst_value = item['hour']

            # Find worst Weekday
            for item in stats_w:
                if item['trades'] == 0: continue
                profit = item['total_profit']
                if profit < 0:
                    metric = abs(profit) * 1000
                    if metric > worst_metric:
                        worst_metric = metric
                        worst_type = 'weekday'
                        worst_value = item['weekday']

            # Second Pass: If no negative profits, look for lowest Sharpe
            if worst_type is None:
                min_sharpe = float('inf')
                
                for item in stats_h:
                    if item['trades'] == 0: continue
                    if item['sharpe_ratio'] < min_sharpe:
                        min_sharpe = item['sharpe_ratio']
                        worst_type = 'hour'
                        worst_value = item['hour']
                
                for item in stats_w:
                    if item['trades'] == 0: continue
                    if item['sharpe_ratio'] < min_sharpe:
                        min_sharpe = item['sharpe_ratio']
                        worst_type = 'weekday'
                        worst_value = item['weekday']

            if worst_type is None:
                print("No more optimization possible.")
                break
                
            # Remove the worst element
            if worst_type == 'hour':
                if len(current_filters['hours']) <= 2: break # Don't remove all
                if worst_value in current_filters['hours']:
                    print(f"Removing Hour {worst_value} (Profit/Sharpe low)")
                    current_filters['hours'].remove(worst_value)
                else:
                    print(f"Warning: Hour {worst_value} not in filters. Stopping loop to avoid infinite.")
                    break
            else:
                if len(current_filters['weekdays']) <= 1: break # Don't remove all
                if worst_value in current_filters['weekdays']:
                    print(f"Removing Weekday {worst_value} (Profit/Sharpe low)")
                    current_filters['weekdays'].remove(worst_value)
                else:
                    print(f"Warning: Weekday {worst_value} not in filters. Stopping loop to avoid infinite.")
                    break
                
            # Re-Apply and Check
            self.analyzer.apply_filters(current_filters)
            summary = self.analyzer.get_summary(use_filtered=True)
            sharpe = summary.get('sharpe_ratio', 0)
            trades = summary.get('total_trades', 0)
            
            print(f"  -> New Sharpe: {sharpe:.2f}, Profit: {summary.get('total_profit_points', 0):.2f}, Trades: {trades}")
            
            # Track Best
            if sharpe > best_sharpe and trades >= 30:
                best_sharpe = sharpe
                best_summary = summary
                # Deep copy filters
                import copy
                best_filters = copy.deepcopy(current_filters)
            
            if trades < 30:
                print("Too few trades. Stopping.")
                break
                
            if sharpe >= TARGET_SHARPE:
                print(f"Target met! Sharpe {sharpe:.2f}")
                self._save_result(current_filters, summary, mode_label)
                return

        # End of Loop - Save Best
        if best_filters:
            print(f"Loop finished. Saving best result: Sharpe {best_sharpe:.2f}")
            self._save_result(best_filters, best_summary, mode_label)
        else:
            print("No improvement found.")

    def _save_result(self, filters, summary, mode_label):
        name = f"Opt {self.strategy_type} {mode_label} (S:{summary['sharpe_ratio']:.1f})"
        config = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": f"Auto-tuned {mode_label}. Sharpe: {summary['sharpe_ratio']}. Profit: {summary['total_profit_points']}",
            "strategy_type": self.strategy_type,
            "created_at": datetime.now().isoformat(),
            "ratio_label": self.analyzer.AVAILABLE_RATIOS[2]['label'], # Use default 1:2
            "filters": filters
        }
        self.save_strategy(config)

if __name__ == "__main__":
    optimizer = StrategyOptimizer('bull_bear_bear')
    optimizer.optimize(inverted=False)
    optimizer.optimize(inverted=True)
