from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import numpy as np

def check_angles():
    # Use existing analyzer logic
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt')
    
    # Try loading cache v4 (from our last run)
    if not analyzer.load_cache():
        print("Cache not found, running simple backtest to generate data...")
        analyzer.load_data()
        analyzer.run_backtest()
        
    df = analyzer.all_trades_df
    print(f"Total trades: {len(df)}")
    
    if 'slope_degrees' not in df.columns:
        print("slope_degrees column MISSING!")
        return

    # Check for zeros
    zeros = df[df['slope_degrees'] == 0]
    print(f"Trades with 0.0 angle: {len(zeros)}")
    
    # Check stats
    print("\nAngle Statistics:")
    print(df['slope_degrees'].describe())
    
    # Check sample of recent trades (tail)
    print("\nRecent Trades Sample (Raw):")
    print(df.tail(5)[['entry_time', 'slope_degrees']])
    
    # Check astype(str) conversion behavior
    recent_str = df.tail(5)[['entry_time', 'slope_degrees']].astype(str).to_dict('records')
    print("\nRecent Trades Sample (After astype(str)):")
    for r in recent_str:
        print(r)

    print("\nWeekday Distribution in TRADES:")
    print(df['weekday'].value_counts().sort_index())
    
    wed_trades = df[df['weekday'] == 2]
    print(f"\nTrades on Wednesday (weekday=2): {len(wed_trades)}")
    if not wed_trades.empty:
        print("Sample Wednesday Trade:")
        print(wed_trades.iloc[0][['entry_time', 'weekday', 'result']])


if __name__ == "__main__":
    check_angles()
