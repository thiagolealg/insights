from strategy_analyzer import StrategyAnalyzer
import pandas as pd

def diagnose():
    print("Initializing StrategyAnalyzer...")
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt')
    
    print("Loading data...")
    analyzer.load_data()
    print(f"Total rows: {len(analyzer.df)}")
    
    # Check 18h
    rows_18h = analyzer.df[analyzer.df['hour'] == 18]
    print(f"\nRows at 18h: {len(rows_18h)}")
    
    # Check Wednesday (weekday=2)
    rows_wed = analyzer.df[analyzer.df['weekday'] == 2]
    print(f"Rows on Wednesday: {len(rows_wed)}")
    
    # Check Signals
    print("\nSimulating signals...")
    analyzer.signals = []
    # Simplified check loop
    i = 20
    length = len(analyzer.df)
    
    # We only care about finding ANY signal at these times to prove capability
    # We'll use the check_buy_conditions from the class (which now has filters removed)
    
    signals_18h = 0
    signals_wed = 0
    
    search_limit = min(length, 500000) # Check first 500k rows quickly or full if fast enough
    search_limit = length
    
    # Use a simpler iteration or just sample? 
    # Let's iterate. Python is slow but acceptable for diagnosis.
    
    for i in range(20, search_limit):
        # Only check if relevant time
        h = analyzer.df['hour'].values[i]
        w = analyzer.df['weekday'].values[i]
        
        if h != 18 and w != 2:
            continue
            
        is_buy = analyzer.check_buy_conditions(i)
        is_sell = analyzer.check_sell_conditions(i)
        
        if is_buy or is_sell:
            if h == 18: signals_18h += 1
            if w == 2: signals_wed += 1
            
    print(f"\nSignals found at 18h: {signals_18h}")
    print(f"Signals found on Wednesday: {signals_wed}")

if __name__ == "__main__":
    diagnose()
