
from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import os

def validate_counts():
    data_file = 'attached_assets/win_1767085916180.txt'
    
    print("=== Validating Signal Counts (V11 Logic) ===\n")
    
    # 1. Inside Bar
    print("1. Strategy: Inside Bar")
    analyzer_ib = StrategyAnalyzer(data_file)
    analyzer_ib.strategy_type = 'inside_bar'
    analyzer_ib.load_data()
    # Force load from cache if exists to verify what App sees
    loaded = analyzer_ib.load_cache()
    if loaded:
        print(f"   [CACHE LOADED] Cache file found.")
    else:
        print(f"   [CALCULATING] Cache not found, running backtest...")
        analyzer_ib.run_backtest()
        
    trades_ib = analyzer_ib.all_trades_df
    count_ib = len(trades_ib)
    print(f"   > Total Trades: {count_ib}")
    if count_ib > 0:
        winners = trades_ib[trades_ib['winner'] == True].shape[0]
        losers = trades_ib[trades_ib['winner'] == False].shape[0]
        print(f"   > Winners: {winners} | Losers: {losers}")
        print(f"   > Win Rate: {winners/count_ib*100:.2f}%")
        
    print("-" * 30)

    # 2. Micro Channel
    print("2. Strategy: Micro Channel")
    analyzer_mc = StrategyAnalyzer(data_file)
    analyzer_mc.strategy_type = 'micro_channel'
    analyzer_mc.load_data()
    loaded_mc = analyzer_mc.load_cache()
    if loaded_mc:
         print(f"   [CACHE LOADED] Cache file found.")
    else:
         print(f"   [CALCULATING] Cache not found, running backtest...")
         analyzer_mc.run_backtest()
         
    trades_mc = analyzer_mc.all_trades_df
    count_mc = len(trades_mc)
    print(f"   > Total Trades: {count_mc}")
    if count_mc > 0:
        winners = trades_mc[trades_mc['winner'] == True].shape[0]
        losers = trades_mc[trades_mc['winner'] == False].shape[0]
        print(f"   > Winners: {winners} | Losers: {losers}")
        print(f"   > Win Rate: {winners/count_mc*100:.2f}%")

if __name__ == "__main__":
    validate_counts()
