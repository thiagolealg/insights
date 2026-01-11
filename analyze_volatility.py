
from strategy_analyzer import StrategyAnalyzer
import pandas as pd

def analyze_vol():
    data_file = 'attached_assets/win_1767085916180.txt'
    print("=== Analysis: Inside Bar Performance by Volatility (V11) ===\n")
    
    analyzer = StrategyAnalyzer(data_file)
    analyzer.strategy_type = 'inside_bar'
    
    # Load Cache V11
    if analyzer.load_cache():
        print(f"Cache Loaded: {len(analyzer.all_trades_df)} raw trades.")
        # Set default ratio to populate trades_df
        analyzer.set_active_ratio(analyzer.AVAILABLE_RATIOS[2]['label']) # TP 1: SL 2 usually default? Or checking all?
        
        # Check explicit stats
        stats = analyzer.get_stats_by_volatility()
        
        print(f"{'Volatility':<15} | {'Trades':<8} | {'Win Rate':<8} | {'Profit':<10} | {'Sharpe':<8}")
        print("-" * 65)
        
        for s in stats:
             print(f"{s['volatility']:<15} | {s['trades']:<8} | {s['win_rate']:<8}% | {s['total_profit']:<10} | {s['sharpe_ratio']:<8}")
             
    else:
        print("[ERROR] Cache V11 not found. Please run backtest first.")

if __name__ == "__main__":
    analyze_vol()
