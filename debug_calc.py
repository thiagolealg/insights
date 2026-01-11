from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import math
import numpy as np

# Mocking StrategyAnalyzer minimal load
def debug_angle():
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt')
    analyzer.load_data()
    
    # Locate specific timestamp
    target_time = pd.Timestamp('2024-10-03 10:10:00')
    mask = analyzer.df['time'] == target_time
    
    if not mask.any():
        print(f"Time {target_time} not found in data.")
        return

    idx = analyzer.df.index[mask][0]
    print(f"Found index {idx} for time {target_time}")
    
    # Internal variables manually extracted
    sma = analyzer.df['sma20'].values
    atr = analyzer.df['atr'].values
    
    val_sma_i = sma[idx]
    val_sma_prev = sma[idx-5]
    val_atr = atr[idx]
    
    dy = val_sma_i - val_sma_prev
    avg_atr = val_atr if not pd.isna(val_atr) and val_atr > 0 else 1.0
    
    raw_slope = dy / 5
    norm_slope = (raw_slope / avg_atr)
    scaled_slope = norm_slope * 10
    angle = math.degrees(math.atan(scaled_slope))
    
    print(f"SMA[i]: {val_sma_i}")
    print(f"SMA[i-5]: {val_sma_prev}")
    print(f"Diff (dy): {dy}")
    print(f"ATR: {val_atr}")
    print(f"Raw Slope (pts/bar): {raw_slope}")
    print(f"Norm Slope (atr/bar): {norm_slope}")
    print(f"Scaled Slope (*10): {scaled_slope}")
    print(f"Calculated Angle: {angle}")
    
    # Call method directly
    method_angle = analyzer.calculate_angle(idx)
    print(f"Method Result: {method_angle}")

if __name__ == "__main__":
    debug_angle()
