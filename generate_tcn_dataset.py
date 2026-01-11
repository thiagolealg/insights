
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
WINDOW_SIZE = 120  # 2 Hours (Manus Recommendation)
FUTURE_LOOKAHEAD = 10  # Check breakout within 10 candles
BREAKOUT_THRESHOLD = 20.0 # Points above/below mother bar
TARGET_CLASS_COUNT = 5000 

def calculate_tcn_features(df):
    """
    Calculates 25+ features for TCN as recommended by Manus AI.
    Focus: Volatility Compression & Momentum.
    """
    df = df.copy()
    
    # 1. Price Basics (Normalized)
    df['ret'] = df['close'].pct_change()
    df['ret_log'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Volatility Compression (The Key)
    # BB Width
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * df['std20']
    df['bb_lower'] = df['sma20'] - 2 * df['std20']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma20'] + 1e-9)
    
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_norm'] = df['atr'] / (df['close'] + 1e-9)
    
    # Range Ratio (Current vs Prev)
    df['range'] = df['high'] - df['low']
    df['range_ratio'] = df['range'] / (df['range'].shift(1) + 1e-9)
    
    # 3. Momentum
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = df['rsi'] / 100.0
    
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ADX (Trend Strength) - Simplified
    # (Omitted for brevity, using BBWidth as proxy for non-trend)
    
    # 4. Volume
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
    # Normalize OBV (Slope over window?) -> OBV ROC
    df['obv_roc'] = df['obv'].pct_change(periods=5)
    
    # Volume SMA Relative
    df['vol_sma'] = df['tick_volume'].rolling(20).mean()
    df['vol_rel'] = df['tick_volume'] / (df['vol_sma'] + 1e-9)
    
    # 5. Time Encoding (Cyclical)
    # Assumes 'time' column is present or we infer from index if needed.
    # Currently dataset doesn't have parsed time. We'll use localized index proxy or mock it.
    # Actually 'active_win*.txt' usually has Time strings.
    if 'time' in df.columns:
        try:
            # Try parsing logic. If slow, skip or use rough modulo
             # Just Mocking Periodicity for now (Minutes in Day)
             # df['minute_sin'] ...
             pass
        except:
             pass

    # Feature List
    features = ['ret_log', 'bb_width', 'atr_norm', 'range_ratio', 'rsi_norm', 
                'macd', 'macd_hist', 'obv_roc', 'vol_rel']
                
    # Fill NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df, features

def label_inside_bar_breakouts(df):
    """
    Labels for TCN:
    0: Noise
    1: Breakout Up (Profitable)
    2: Breakout Down (Profitable)
    
    Logic:
    We look at 'Inside Bars' only? No, TCN runs on all windows, but we target breakouts.
    Actually, Manus said "Trading Inside Bar Breakouts".
    So we should filtering samples where the *Trigger* (last candle of window) IS an Inside Bar.
    """
    df['label'] = 0
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    labels = np.zeros(len(df), dtype=int)
    
    # Scan
    for i in range(1, len(df) - FUTURE_LOOKAHEAD):
        # 1. Check Inside Bar Condition at i
        # Current High < Prev High AND Current Low > Prev Low
        inside = (highs[i] <= highs[i-1]) and (lows[i] >= lows[i-1])
        
        if not inside:
            continue
            
        # 2. Check Future Breakout
        mother_high = highs[i-1]
        mother_low = lows[i-1]
        
        # Look ahead 
        future_highs = highs[i+1 : i+1+FUTURE_LOOKAHEAD]
        future_lows = lows[i+1 : i+1+FUTURE_LOOKAHEAD]
        
        break_up = np.any(future_highs > mother_high + BREAKOUT_THRESHOLD)
        break_down = np.any(future_lows < mother_low - BREAKOUT_THRESHOLD)
        
        if break_up and not break_down:
            labels[i] = 1
        elif break_down and not break_up:
            labels[i] = 2
        else:
            labels[i] = 0 # Choppy or no breakout
            
    df['label'] = labels
    return df

def generate_tcn_dataset():
    print("--- Generating TCN Dataset (Manus Plan) ---")
    data_file = "attached_assets/win_1767085916180.txt"
    if not os.path.exists(data_file):
        print("Data file not found.")
        return

    df = pd.read_csv(data_file)
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Featurize
    df, feature_cols = calculate_tcn_features(df)
    
    # 2. Label
    df = label_inside_bar_breakouts(df)
    
    # 3. Create Windows
    # Z-Score Normalization of Features
    scaler = StandardScaler()
    data_vals = scaler.fit_transform(df[feature_cols].values)
    labels = df['label'].values
    
    X_data = []
    y_data = []
    
    # Select Indices (Only Inside Bars that are Breakouts or Failures)
    # We want to train the model to distinguish Good IB from Bad IB.
    # Class 0: Inside Bar but Fails/Choppy.
    # Class 1: Inside Bar -> Up
    # Class 2: Inside Bar -> Down
    
    # Filter to only indexes that ARE Inside Bars (label logic already did this implicitly? 
    # No, label logic marked indices. those with 0 might be Non-IB OR IB-Fail.
    # We need to explicitly check IB again or check if we want to train on ALL candles.)
    # Manus: "Trading Inside Bar Breakouts". Model likely receives an Inside Bar context.
    
    # Let's filter for Inside Bars again to be sure we only train on them.
    # (Optimization: We could have saved 'is_inside' column).
    
    highs = df['high'].values
    lows = df['low'].values
    
    idxs_1 = []
    idxs_2 = []
    idxs_0 = []
    
    for i in range(WINDOW_SIZE, len(df) - FUTURE_LOOKAHEAD):
        is_inside = (highs[i] <= highs[i-1]) and (lows[i] >= lows[i-1])
        if not is_inside:
            continue
            
        lbl = labels[i]
        
        if lbl == 1: idxs_1.append(i)
        elif lbl == 2: idxs_2.append(i)
        else: idxs_0.append(i)

    print(f"Inside Bars Found: Up={len(idxs_1)}, Down={len(idxs_2)}, Fail/Hold={len(idxs_0)}")
    
    # Balance
    n_min = min(len(idxs_1), len(idxs_2), TARGET_CLASS_COUNT)
    # If 0 count is huge, downsample.
    n_0 = min(len(idxs_0), n_min * 2) 
    
    np.random.shuffle(idxs_1)
    np.random.shuffle(idxs_2)
    np.random.shuffle(idxs_0)
    
    use_idxs = np.concatenate([idxs_1[:n_min], idxs_2[:n_min], idxs_0[:n_0]])
    np.random.shuffle(use_idxs)
    
    print(f"Using {len(use_idxs)} samples for training.")
    
    for i in use_idxs:
        # Window: [i - Window + 1 : i + 1]
        window = data_vals[i-WINDOW_SIZE+1 : i+1]
        if window.shape != (WINDOW_SIZE, len(feature_cols)):
            continue
            
        # TCN Input: (Batch, Channels, Length) or (Batch, Length, Channels)?
        # PyTorch Conv1d expects (Batch, Channels, Length).
        # Our window is (Length, Channels). Transpose it.
        window = window.T # (Channels, Length)
        
        X_data.append(window)
        y_data.append(labels[i])
        
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.longlong)
    
    print(f"Final Dataset: {X_data.shape}")
    
    torch.save(torch.from_numpy(X_data), 'train_inputs_tcn.pt')
    torch.save(torch.from_numpy(y_data), 'train_labels_tcn.pt')
    print("Saved TCN dataset.")

if __name__ == "__main__":
    generate_tcn_dataset()
