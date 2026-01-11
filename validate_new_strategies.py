
from strategy_analyzer import StrategyAnalyzer
import pandas as pd

def validate_new():
    data_file = 'attached_assets/win_1767085916180.txt'
    strategies = [
        'bull_bear', 
        'bull_bear_bear', 
        'sequence_reversal', 
        'sma_trend'
    ]
    
    print("=== Validating New Strategies ===\n")
    print(f"{'Strategy':<20} | {'Trades':<8} | {'Win Rate':<8}")
    print("-" * 50)
    
    for strats in strategies:
        analyzer = StrategyAnalyzer(data_file)
        analyzer.strategy_type = strats
        analyzer.load_data()
        
        # Determine signals
        # find_signals (called by run_backtest) handles dispatch based on strategy_type
        
        # Force run backtest
        analyzer.run_backtest()
        
        # Get Stats
        stats = analyzer.get_summary()
        wr = stats['win_rate'] if stats['total_trades'] > 0 else 0
        
        print(f"{strats:<20} | {stats['total_trades']:<8} | {wr:<8}%")

if __name__ == "__main__":
    validate_new()
