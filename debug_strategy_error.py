
import traceback
from strategy_analyzer import StrategyAnalyzer

def debug_strat(strat_name):
    print(f"=== Debugging {strat_name} ===")
    try:
        data_file = 'attached_assets/win_1767085916180.txt'
        analyzer = StrategyAnalyzer(data_file)
        analyzer.strategy_type = strat_name
        
        print("Loading Data...")
        analyzer.load_data()
        
        print("Running Backtest...")
        # Force fresh backtest to trigger logic
        analyzer.run_backtest()
        
        print("Getting Summary...")
        stats = analyzer.get_summary()
        print("Success!")
        print(stats)
        
    except Exception:
        print(f"FAILED {strat_name}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_strat('bull_bear')
    print("\n" + "="*30 + "\n")
    debug_strat('sequence_reversal')
