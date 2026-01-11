from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import numpy as np

try:
    print("Initializing Analyzer...")
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt')
    print("Loading Data...")
    analyzer.load_data()
    print(f"Data loaded. Rows: {len(analyzer.df)}")
    print(f"SMA non-na count: {analyzer.df['sma20'].count()}")
    print(f"ATR non-na count: {analyzer.df['atr'].count()}")

    print("Counting Signals...")
    signals = []
    i = 20
    length = len(analyzer.df)
    
    # Simple loop simulation (copied logic)
    while i < length - 1:
        is_buy = analyzer.check_buy_conditions(i)
        is_sell = analyzer.check_sell_conditions(i)
        
        if is_buy:
            signals.append(i)
            i += 1
        elif is_sell:
            signals.append(i)
            i += 1
        else:
            i += 1
            
    print(f"Total Signals Found: {len(signals)}")
    
    print("Testing Angle Calculation on Signals...")
    errors = 0
    for idx in signals[:1000]: # Test first 1000
        try:
            ang = analyzer.calculate_angle(idx)
            if np.isnan(ang):
                print(f"NaN Angle at {idx}")
                errors += 1
        except Exception as e:
            print(f"Error at {idx}: {e}")
            errors += 1
            
    print(f"Angle Calculation Errors: {errors}")
    
    # Load cache if exists to compare
    if analyzer.load_cache():
        print(f"Cache Loaded. Total All Trades: {len(analyzer.all_trades_df)}")
        # Check unique signals in cache
        if not analyzer.all_trades_df.empty:
            unique_entries = analyzer.all_trades_df['entry_idx'].nunique()
            print(f"Unique Entry Indices in Cache: {unique_entries}")
    else:
        print("Cache not found.")

except Exception as e:
    print(f"Fatal Error: {e}")
