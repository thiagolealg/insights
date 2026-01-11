import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from strategy_analyzer import StrategyAnalyzer

# --- MLP Model Definition (Must match trained model) ---
class TradeMLP(nn.Module):
    def __init__(self):
        super(TradeMLP, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU(0.1)
        
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x

def extract_features(df, i, invert=False):
    # MLP Feature Extraction
    if i < 40: return None
    row = df.iloc[i]
    if 'rsi' not in df.columns: return None
    
    rsi = row['rsi']
    slope = row['sma_slope']
    vol = row['vol_ratio']
    dist = row['dist_sma']
    body = row['body_pct']
    wick_u = row['wick_u_pct']
    wick_l = row['wick_l_pct']
    
    if invert:
        rsi = 100.0 - rsi
        slope = -slope
        dist = -dist
        tmp = wick_u; wick_u = wick_l; wick_l = tmp
        
    f_rsi = (rsi - 50.0) / 50.0 
    f_slope = np.clip(slope / 5.0, -1.0, 1.0)
    f_vol = np.clip(vol - 1.0, -1.0, 1.0)
    f_dist = np.clip(dist / 5.0, -1.0, 1.0)
    f_body = (body - 0.5) * 2.0
    f_wik_u = (wick_u - 0.5) * 2.0
    f_wik_l = (wick_l - 0.5) * 2.0
    
    features = np.array([f_rsi, f_slope, f_vol, f_dist, f_body, f_wik_u, f_wik_l], dtype=np.float32)
    return features

def generate_validation_charts():
    print("--- Generating AI Approved Charts ---")
    output_dir = "validation_charts"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Load Model
    device = torch.device("cpu")
    model = TradeMLP().to(device)
    if os.path.exists("trade_mlp.pth"):
        model.load_state_dict(torch.load("trade_mlp.pth", map_location=device, weights_only=False))
        model.eval()
        print("MLP Model loaded.")
    else:
        print("Model trade_mlp.pth not found!")
        return

    data_file = "attached_assets/win_1767085916180.txt"
    strategy = 'bull_bear'
    
    analyzer = StrategyAnalyzer(data_file, strategy_type=strategy)
    analyzer.run_backtest()
    if analyzer.df is None: analyzer.load_data()
    
    # Enhance DF
    df = analyzer.df
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma_slope'] = df['sma20'].diff()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['std_dev'] = df['close'].rolling(window=20).std()
    vol_long = df['std_dev'].rolling(window=100).mean()
    df['vol_ratio'] = df['std_dev'] / (vol_long + 1e-9)
    df['dist_sma'] = (df['close'] - df['sma20']) / (df['sma20'] + 1e-9) * 1000.0
    candle_range = df['high'] - df['low']
    df['body_pct'] = (df['close'] - df['open']).abs() / (candle_range + 1e-9)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['wick_u_pct'] = df['upper_wick'] / (candle_range + 1e-9)
    df['wick_l_pct'] = df['lower_wick'] / (candle_range + 1e-9)
    
    trades = analyzer.trades
    print(f"Scanning {len(trades)} trades for AI Approval...")
    
    approved_trades = []
    
    for trade in trades:
        idx = trade['entry_idx']
        is_sell = (trade['direction'] == 'sell')
        
        feats = extract_features(df, idx, invert=is_sell)
        if feats is None: continue
        
        t_x = torch.tensor(feats).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t_x)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
            
        if pred.item() == 1 and conf.item() > 0.60:
            trade['ai_conf'] = conf.item()
            approved_trades.append(trade)
            
    print(f"AI Approved {len(approved_trades)} trades.")
    
    # Plot top 50
    # Sort by confidence
    approved_trades.sort(key=lambda x: x['ai_conf'], reverse=True)
    to_plot = approved_trades[:50]
    
    for i, trade in enumerate(to_plot):
        idx = trade['entry_idx']
        
        # Plot Logic (Simplified)
        start = max(0, idx - 30); end = min(len(df), idx + 20)
        w = df.iloc[start:end].reset_index(drop=True)
        w['x'] = range(len(w))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        up = w[w.close >= w.open]
        down = w[w.close < w.open]
        
        ax.bar(up.x, up.close-up.open, 0.6, up.open, color='green')
        ax.bar(up.x, up.high-up.close, 0.1, up.close, color='green')
        ax.bar(up.x, up.low-up.open, 0.1, up.open, color='green')
        
        ax.bar(down.x, down.close-down.open, 0.6, down.open, color='red')
        ax.bar(down.x, down.high-down.open, 0.1, down.open, color='red')
        ax.bar(down.x, down.low-down.close, 0.1, down.close, color='red')
        
        ax.plot(w.x, w.sma20, color='blue', label='SMA20')
        
        # Entry Marker
        entry_x = idx - start
        ax.axvline(entry_x, color='black', linestyle='--')
        
        res = "WIN" if trade['result'] > 0 else "LOSS"
        conf_pct = trade['ai_conf'] * 100
        ax.set_title(f"AI APPROVED ({conf_pct:.1f}%) - {trade['direction'].upper()} - {res}")
        
        plt.savefig(f"{output_dir}/ai_chart_{i+1}_{res}.png")
        plt.close()
        
    print("Charts generated.")

if __name__ == "__main__":
    generate_validation_charts()
