from strategy_analyzer import StrategyAnalyzer
import pandas as pd

def check_signals():
    print("Initializing StrategyAnalyzer...")
    # Use the file found in attached_assets
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt', strategy_type='inside_bar')
    
    print("Loading data...")
    analyzer.load_data()
    print(f"Data loaded: {len(analyzer.df)} rows")
    
    print("Running backtest (signal detection only)...")
    # Manually run the signal detection part to avoid full simulation overhead for this check
    signals = []
    i = 20
    length = len(analyzer.df)
    
    # Intentionally running the raw check methods from the class
    count_buy = 0
    count_sell = 0
    
    while i < length - 1:
        is_buy = analyzer.check_buy_conditions(i)
        is_sell = analyzer.check_sell_conditions(i)
        
        if is_buy:
            signals.append({'idx': i, 'type': 'buy'})
            count_buy += 1
            i += 1
        elif is_sell:
            signals.append({'idx': i, 'type': 'sell'})
            count_sell += 1
            i += 1
        else:
            i += 1
            
    print(f"Total Signals Found: {len(signals)}")
    print(f"Buy Signals: {count_buy}")
    print(f"Sell Signals: {count_sell}")

if __name__ == "__main__":
    check_signals()
