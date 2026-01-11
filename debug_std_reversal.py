
import pandas as pd
import numpy as np
from strategy_analyzer import StrategyAnalyzer

def debug_std():
    analyzer = StrategyAnalyzer("attached_assets/win_1767085916180.txt", strategy_type='std_reversal')
    analyzer.load_data()
    
    # Check Buy Signals
    print("Checking Buy Signals (rev_std_buy_signal)...")
    buy_signals = analyzer.df[analyzer.df['rev_std_buy_signal'] > 3]
    print(f"Total Buy Signals (Count > 3): {len(buy_signals)}")
    if not buy_signals.empty:
        print(buy_signals[['time', 'open', 'high', 'low', 'close', 'rev_std_buy_signal']].head())
        
    # Check Sell Signals
    print("\nChecking Sell Signals (rev_std_sell_signal)...")
    sell_signals = analyzer.df[analyzer.df['rev_std_sell_signal'] > 3]
    print(f"Total Sell Signals (Count > 3): {len(sell_signals)}")
    if not sell_signals.empty:
        print(sell_signals[['time', 'open', 'high', 'low', 'close', 'rev_std_sell_signal']].head())
    
    # Inspect calculation for a few rows
    analyzer.df['std_dev_calc'] = analyzer.df[['high', 'low', 'close']].std(axis=1)
    analyzer.df['hc_diff_1.2'] = (analyzer.df['high'] - analyzer.df['close']) * 1.2
    analyzer.df['cl_diff_1.2'] = (analyzer.df['close'] - analyzer.df['low']) * 1.2
    analyzer.df['cond_buy_manual'] = analyzer.df['std_dev_calc'] > analyzer.df['hc_diff_1.2']
    analyzer.df['cond_sell_manual'] = analyzer.df['std_dev_calc'] > analyzer.df['cl_diff_1.2']
    
    print("\nValidation of first 10 rows:")
    print(analyzer.df[['time', 'std_dev_calc', 'hc_diff_1.2', 'cond_buy_manual', 'cl_diff_1.2', 'cond_sell_manual']].head(10))

if __name__ == "__main__":
    debug_std()
