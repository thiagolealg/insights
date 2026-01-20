import sys
import os
import shutil
import pandas as pd
from strategy_analyzer import StrategyAnalyzer

# Force path to ensure imports work
sys.path.append(os.getcwd())

def check_cache_columns():
    print("Checking cache...")
    analyzer = StrategyAnalyzer('bull_bear')
    # Try loading cache
    if analyzer.load_cache():
        print("Loaded from cache.")
        cols = analyzer.all_trades_df.columns.tolist()
        print("Columns in cache:", cols)
        if 'vol_slope_level' in cols and 'jerk_level' in cols:
            print("SUCCESS: Columns found in cache.")
        else:
            print("FAILURE: New columns NOT found in cache.")
            return False
    else:
        print("No cache found (or load failed).")
        return False
    return True

if __name__ == "__main__":
    if not check_cache_columns():
        print(" Clearing .cache directory to force recalculation...")
        cache_dir = os.path.join(os.getcwd(), '.cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("Cache cleared.")
        else:
            print("Cache directory not found.")
        
        print("Running backtest to regenerate...")
        analyzer = StrategyAnalyzer('bull_bear')
        analyzer.run_backtest()
        cols = analyzer.all_trades_df.columns.tolist()
        if 'vol_slope_level' in cols and 'jerk_level' in cols:
             print("SUCCESS: Columns generated after run.")
             print(analyzer.all_trades_df[['vol_slope_level', 'jerk_level']].head())
        else:
             print("FAILURE: Columns still missing after run.")
