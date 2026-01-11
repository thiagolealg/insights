
from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import numpy as np

class Optimizer(StrategyAnalyzer):
    def __init__(self, data_file, variant='v11_baseline'):
        super().__init__(data_file)
        self.variant = variant
        self.strategy_type = 'inside_bar' # Focus on Inside Bar
        
    def _get_cache_filename(self):
        # Disable cache for optimization to force recalc
        return f".cache/opt_{self.variant}.pkl"

    def check_buy_conditions(self, i):
        if i < 20: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        high = self.df['high'].values
        low = self.df['low'].values
        sma20 = self.df['sma20'].values
        
        # Base Data
        if pd.isna(sma20[i-3]): return False
        
        # V11 Baseline Components
        cond_micro_trend = close[i-3] < close[i-2]
        cond_bullish_context = close[i-3] > open_price[i-3] and close[i-2] > open_price[i-2]
        body_high_i2 = max(open_price[i-2], close[i-2])
        body_low_i2 = min(open_price[i-2], close[i-2])
        cond_inside = (open_price[i-1] >= body_low_i2 and open_price[i-1] <= body_high_i2 and close[i-1] >= body_low_i2 and close[i-1] <= body_high_i2)
        mid_point_i2 = (high[i-2] + low[i-2]) / 2
        cond_above_sma = open_price[i-3] > sma20[i-3] and close[i-3] > sma20[i-3]

        # VARIANT LOGIC
        if self.variant == 'v11_baseline':
            # Strict V11: Low > 50%
            cond_position = low[i-1] > mid_point_i2
            return cond_micro_trend and cond_bullish_context and cond_inside and cond_position and cond_above_sma
            
        elif self.variant == 'v11_contraction':
            # V11 + Inside Range < 50% Mother Range
            cond_position = low[i-1] > mid_point_i2
            range_i1 = high[i-1] - low[i-1]
            range_i2 = high[i-2] - low[i-2]
            cond_contract = range_i1 < (range_i2 * 0.5)
            return cond_micro_trend and cond_bullish_context and cond_inside and cond_position and cond_above_sma and cond_contract

        elif self.variant == 'v11_momentum':
            # V11 + Mother Bar > Grandparent (Acceleration)
            cond_position = low[i-1] > mid_point_i2
            body_i2 = abs(close[i-2] - open_price[i-2])
            body_i3 = abs(close[i-3] - open_price[i-3])
            cond_mom = body_i2 > body_i3
            return cond_micro_trend and cond_bullish_context and cond_inside and cond_position and cond_above_sma and cond_mom

        elif self.variant == 'v11_extreme':
            # Top Third (Low > 66%)
            top_third = low[i-2] + ((high[i-2] - low[i-2]) * 0.66)
            cond_position = low[i-1] > top_third
            return cond_micro_trend and cond_bullish_context and cond_inside and cond_position and cond_above_sma

        return False

    def check_sell_conditions(self, i):
        if i < 20: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        high = self.df['high'].values
        low = self.df['low'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-3]): return False
        
        cond_micro_trend_down = close[i-3] > close[i-2]
        cond_bearish_context = close[i-3] < open_price[i-3] and close[i-2] < open_price[i-2]
        body_high_i2 = max(open_price[i-2], close[i-2])
        body_low_i2 = min(open_price[i-2], close[i-2])
        cond_inside = (open_price[i-1] >= body_low_i2 and open_price[i-1] <= body_high_i2 and close[i-1] >= body_low_i2 and close[i-1] <= body_high_i2)
        mid_point_i2 = (high[i-2] + low[i-2]) / 2
        cond_below_sma = open_price[i-3] < sma20[i-3] and close[i-3] < sma20[i-3]

        if self.variant == 'v11_baseline':
            cond_position = high[i-1] < mid_point_i2
            return cond_micro_trend_down and cond_bearish_context and cond_inside and cond_position and cond_below_sma
            
        elif self.variant == 'v11_contraction':
            cond_position = high[i-1] < mid_point_i2
            range_i1 = high[i-1] - low[i-1]
            range_i2 = high[i-2] - low[i-2]
            cond_contract = range_i1 < (range_i2 * 0.5)
            return cond_micro_trend_down and cond_bearish_context and cond_inside and cond_position and cond_below_sma and cond_contract

        elif self.variant == 'v11_momentum':
            cond_position = high[i-1] < mid_point_i2
            body_i2 = abs(close[i-2] - open_price[i-2])
            body_i3 = abs(close[i-3] - open_price[i-3])
            cond_mom = body_i2 > body_i3
            return cond_micro_trend_down and cond_bearish_context and cond_inside and cond_position and cond_below_sma and cond_mom

        elif self.variant == 'v11_extreme':
            # Bottom Third (High < 33%)
            bottom_third = low[i-2] + ((high[i-2] - low[i-2]) * 0.33)
            cond_position = high[i-1] < bottom_third
            return cond_micro_trend_down and cond_bearish_context and cond_inside and cond_position and cond_below_sma

        return False

def run_optimization():
    data_file = 'attached_assets/win_1767085916180.txt'
    variants = ['v11_baseline', 'v11_contraction', 'v11_momentum', 'v11_extreme']
    
    print(f"{'Variant':<20} | {'Trades':<8} | {'Win Rate':<8} | {'Profit':<10} | {'Sharpe':<8}")
    print("-" * 70)
    
    for v in variants:
        opt = Optimizer(data_file, variant=v)
        opt.run_backtest() # Will run slow but accurate
        
        # Force default filter stats
        opt.set_active_ratio('TP 1x : SL 2x') # Keeping constant
        
        stats = opt.get_summary()
        print(f"{v:<20} | {stats['total_trades']:<8} | {stats['win_rate']:<8}% | {stats['total_profit_points']:<10} | {stats['sharpe_ratio']:<8}")

if __name__ == "__main__":
    run_optimization()
