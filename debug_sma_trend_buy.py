import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('attached_assets/win_1767085916180.txt')
df['datetime'] = pd.to_datetime(df['time'])
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# Calculate indicators manually to ensure match
df['sma20'] = df['close'].rolling(window=20).mean()

close = df['close'].values
open_price = df['open'].values
sma20 = df['sma20'].values
valid_count = 0

print(f"Total rows: {len(df)}")
for i in range(20, len(df)):
    if pd.isna(sma20[i-2]): continue
    
    cond_i2_above_sma = close[i-2] > sma20[i-2]
    cond_i1_bull = close[i-1] > open_price[i-1]
    
    if cond_i2_above_sma and cond_i1_bull:
        valid_count += 1
        if valid_count <= 5:
            print(f"Found Buy Signal at index {i}: Date={df.index[i]}, Close[i-2]={close[i-2]}, SMA[i-2]={sma20[i-2]}, Candle[i-1]={'Bull' if cond_i1_bull else 'Bear'}")

print(f"Total Buy Signals Found: {valid_count}")
