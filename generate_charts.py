import os
import random
import pandas as pd
import mplfinance as mpf
from strategy_analyzer import StrategyAnalyzer

def generate_validation_charts():
    # Setup directories
    output_dir = "validation_charts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Initialize analyzer
    data_file = "attached_assets/win_1767085916180.txt"
    print(f"Loading data from {data_file}...")
    analyzer = StrategyAnalyzer(data_file)
    analyzer.run_backtest()
    
    trades = analyzer.trades
    print(f"Total trades found: {len(trades)}")
    
    buys = [t for t in trades if t['direction'] == 'Compra']
    sells = [t for t in trades if t['direction'] == 'Venda']
    
    print(f"Found {len(buys)} buys and {len(sells)} sells.")
    
    # Sample 50 of each
    sampled_buys = random.sample(buys, min(50, len(buys)))
    sampled_sells = random.sample(sells, min(50, len(sells)))
    
    all_sampled = [('BUY', t) for t in sampled_buys] + [('SELL', t) for t in sampled_sells]
    
    print(f"Generating {len(all_sampled)} charts...")
    
    # Use the full dataframe from analyzer
    df = analyzer.df.copy()
    df.set_index('time', inplace=True)
    
    for i, (label, trade) in enumerate(all_sampled):
        idx = trade['entry_idx']
        outcome = "GAIN" if trade['winner'] else "LOSS"
        result_pts = trade['result']
        
        # Window of 60 candles: 30 before, 30 after
        start_idx = max(0, idx - 30)
        end_idx = min(len(df), idx + 31)
        
        window_df = df.iloc[start_idx:end_idx].copy()
        
        # Signal time for marking on chart
        signal_time = trade['entry_time']
        
        # Create additional plot for SMA20
        ap = mpf.make_addplot(window_df['sma20'], color='blue', width=1.5)
        
        # Define filename
        filename = f"{output_dir}/{outcome}_{label}_{i+1}_{signal_time.strftime('%Y%m%d_%H%M')}.png"
        
        # Plot with Target and Stop lines
        try:
            angle = trade.get('slope_degrees', 0)
            mpf.plot(window_df, 
                     type='candle', 
                     style='charles',
                     title=f"Trend: {angle:.1f}° | Profit: {result_pts} pts | {signal_time}",
                     ylabel='Price',
                     addplot=ap,
                     savefig=filename,
                     vlines=dict(vlines=signal_time, colors='gray', linestyle='--'),
                     hlines=dict(hlines=[trade['target_price'], trade['stop_price']], 
                                 colors=['g', 'r'], linestyle='-', linewidths=1.5),
                     tight_layout=True)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i+1}/{len(all_sampled)} charts...")
        except Exception as e:
            print(f"Error generating chart {i+1}: {e}")

    print(f"Finished. Charts saved to {output_dir}/")

if __name__ == "__main__":
    generate_validation_charts()
