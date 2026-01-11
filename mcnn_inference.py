
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from generate_mcnn_dataset import calculate_technical_indicators
from train_mcnn import TradeMCNN
import os
from strategy_analyzer import StrategyAnalyzer

def run_mcnn_inference():
    print("--- MCNN Inference ---")
    
    model_path = "trade_mcnn.pth"
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    device = torch.device("cpu")
    model = TradeMCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print("MCNN Model Loaded.")
    
    # Load Data
    data_file = "attached_assets/win_1767085916180.txt"
    df = pd.read_csv(data_file)
    df.columns = [c.lower() for c in df.columns]
    
    # Features
    df = calculate_technical_indicators(df)
    
    # Pre-scaling (Must match Generator)
    df['scp'] = np.tanh(df['scp'] * 100)
    df['rsi'] = df['rsi'] / 100.0
    df['wr'] = (df['wr'] + 100) / 100.0
    df['mfi'] = df['mfi'] / 100.0
    macd_mean = df['macd'].mean() 
    macd_std = df['macd'].std()
    df['macd'] = np.clip((df['macd'] - macd_mean) / (macd_std + 1e-9), -3, 3) / 3.0
    
    feat_cols = ['scp', 'rsi', 'wr', 'mfi', 'macd']
    data_matrix = df[feat_cols].values
    
    # Scan
    predictions = []
    confidences = []
    indices = []
    
    print("Running Inference...")
    WINDOW_SIZE = 11
    
    # Batch Processing
    BATCH_SIZE = 10000
    total_len = len(df)
    
    for start_i in range(WINDOW_SIZE, total_len, BATCH_SIZE):
        end_i = min(start_i + BATCH_SIZE, total_len)
        
        batch_X = []
        batch_indices = []
        
        for i in range(start_i, end_i):
            window = data_matrix[i-WINDOW_SIZE+1 : i+1]
            if window.shape != (WINDOW_SIZE, 5): continue
            
            # (11, 5) -> (5, 11)
            batch_X.append(window.T)
            batch_indices.append(i)
            
        if not batch_X: continue
        
        X_tensor = torch.tensor(np.array(batch_X), dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            full_out = model(X_tensor)
            probs = torch.softmax(full_out, dim=1)
            confs, preds = torch.max(probs, 1)
            
            predictions.extend(preds.numpy())
            confidences.extend(confs.numpy())
            indices.extend(batch_indices)
            
    # Analyze
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    indices = np.array(indices)
    
    # Threshold
    THRESHOLD = 0.80 # Paper Goal
    
    # 0=Hold, 1=Buy, 2=Sell
    buy_mask = (predictions == 1) & (confidences >= THRESHOLD)
    sell_mask = (predictions == 2) & (confidences >= THRESHOLD)
    
    buys = indices[buy_mask]
    sells = indices[sell_mask]
    
    # Filter by Inside Bar
    # Logic: High[i] < High[i-1] and Low[i] > Low[i-1]
    
    final_buys = []
    final_sells = []
    
    highs = df['high'].values
    lows = df['low'].values
    
    print("Filtering predictions for Inside Bar context...")
    
    for idx in buys:
        if idx < 1: continue
        is_inside = (highs[idx] <= highs[idx-1]) and (lows[idx] >= lows[idx-1])
        if is_inside:
            final_buys.append(idx)
            
    for idx in sells:
        if idx < 1: continue
        is_inside = (highs[idx] <= highs[idx-1]) and (lows[idx] >= lows[idx-1])
        if is_inside:
            final_sells.append(idx)
            
    print(f"MCNN + Inside Bar: {len(final_buys)} Buys, {len(final_sells)} Sells")
    
    # Calc Stats
    buys = np.array(final_buys)
    sells = np.array(final_sells)
    
    win_b = 0
    tot_b = 0
    pnls = []
    
    closes = df['close'].values
    
    LOOKAHEAD = 5
    
    for idx in buys:
        if idx + LOOKAHEAD >= total_len: continue
        entry = closes[idx]
        exit_p = closes[idx+LOOKAHEAD]
        if exit_p > entry: win_b += 1
        tot_b += 1
        pnls.append(exit_p - entry)
        
    win_s = 0
    tot_s = 0
    for idx in sells:
        if idx + LOOKAHEAD >= total_len: continue
        entry = closes[idx]
        exit_p = closes[idx+LOOKAHEAD]
        if exit_p < entry: win_s += 1
        tot_s += 1
        pnls.append(entry - exit_p)
        
    total_trades = tot_b + tot_s
    total_wins = win_b + win_s
    
    if total_trades > 0:
        wr = 100.0 * total_wins / total_trades
        print(f"--- Performance ({LOOKAHEAD} candles hold) ---")
        print(f"Win Rate: {wr:.2f}%")
        print(f"Total PnL: {sum(pnls):.2f}")
    else:
        print("No trades selected.")

if __name__ == "__main__":
    run_mcnn_inference()
