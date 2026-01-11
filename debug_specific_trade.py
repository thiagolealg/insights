from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import numpy as np

def debug_specific():
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt')
    if not analyzer.load_cache():
        print("Cache not found, running backtest...")
        analyzer.load_data()
        analyzer.run_backtest()
    else:
        # We need data loaded to check SMA values even if results are cached
        analyzer.load_data() 

    df = analyzer.all_trades_df
    target_time = '2024-10-15 17:19:00'
    
    # Filter trade
    trade = df[df['entry_time'].astype(str) == target_time]
    
    if trade.empty:
        print(f"Trade at {target_time} NOT FOUND in all_trades_df")
        return

    print(f"Trade found:\n{trade.iloc[0]}")
    
    entry_idx = trade.iloc[0]['entry_idx']
    print(f"\nRe-calculating angle for index {entry_idx}...")
    
    # Manually calc angle
    sma = analyzer.df['sma20'].values
    atr = analyzer.df['atr'].values
    
    i = entry_idx
    val_sma_i = sma[i]
    val_sma_prev = sma[i-5]
    val_atr = atr[i]
    
    dy = val_sma_i - val_sma_prev
    avg_atr = val_atr if not pd.isna(val_atr) and val_atr > 0 else 1.0
    
    raw_slope = dy / 5
    norm_slope = (raw_slope / avg_atr)
    scaled_slope = norm_slope * 10
    import math
    angle = math.degrees(math.atan(scaled_slope))
    
    output = []
    output.append(f"SMA[i]: {val_sma_i}")
    output.append(f"SMA[i-5]: {val_sma_prev}")
    output.append(f"Diff (dy): {dy}")
    output.append(f"ATR: {val_atr}")
    output.append(f"Calculated Angle: {angle}")
    
    with open('debug_result.txt', 'w') as f:
        f.write('\n'.join(output))
    print("Debug written to debug_result.txt")

if __name__ == "__main__":
    debug_specific()
