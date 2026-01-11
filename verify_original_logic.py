
from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import numpy as np

def verify_original_counts():
    data_file = 'attached_assets/win_1767085916180.txt'
    
    print("=== Validating User's Original Logic ===\n")
    print("Conditions:")
    print("  Buy: close[i-3] < close[i-2] > close[i-1] AND close[i-3] > SMA(20) AND Inside Body")
    print("  Sell: close[i-3] > close[i-2] < close[i-1] AND close[i-3] < SMA(20) AND Inside Body")
    
    analyzer = StrategyAnalyzer(data_file)
    analyzer.load_data()
    
    df = analyzer.df
    close = df['close'].values
    open_price = df['open'].values
    sma20 = df['sma20'].values
    
    # Pre-calculate body stats for inside check
    body_high = np.maximum(open_price, close)
    body_low = np.minimum(open_price, close)
    
    signals = []
    
    print("Scanning...")
    for i in range(20, len(df)):
        # Common Inside Body Check
        # i-1 inside i-2
        is_inside = (open_price[i-1] >= body_low[i-2] and 
                     open_price[i-1] <= body_high[i-2] and 
                     close[i-1] >= body_low[i-2] and 
                     close[i-1] <= body_high[i-2])
                     
        if not is_inside:
            continue
            
        # Buy Logic
        # close[i-3] < close[i-2] > close[i-1]
        # close[i-3] > SMA(20)
        if (close[i-3] < close[i-2] and close[i-2] > close[i-1]):
            if close[i-3] > sma20[i-3]:
                 signals.append('Buy')
                 continue

        # Sell Logic
        # close[i-3] > close[i-2] < close[i-1]
        # close[i-3] < SMA(20) 
        if (close[i-3] > close[i-2] and close[i-2] < close[i-1]):
            if close[i-3] < sma20[i-3]:
                signals.append('Sell')
                continue
                
    count = len(signals)
    print(f"\nTotal Signals Found: {count}")
    print(f"Expected ~200,000?")

if __name__ == "__main__":
    verify_original_counts()
