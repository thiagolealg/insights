from strategy_analyzer import StrategyAnalyzer
import pandas as pd

def check_recent_trades():
    print("Initializing bull_bear strategy with consolidated data...")
    a = StrategyAnalyzer("win_full_data.parquet", strategy_type="bull_bear")
    a.run_backtest()
    
    df = a.all_trades_df
    if df.empty:
        print("No trades found at all.")
        return
        
    years = df['year'].unique()
    print(f"Years with trades: {sorted(years.tolist())}")
    
    recent = df[df['year'] >= 2025]
    print(f"Number of trades in 2025/2026: {len(recent)}")
    if len(recent) > 0:
        print(recent[['entry_time', 'direction', 'result']].head())

if __name__ == "__main__":
    check_recent_trades()
