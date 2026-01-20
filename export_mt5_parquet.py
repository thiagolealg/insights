import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os

def export_mt5_data_to_parquet(symbol="WIN$", timeframe=mt5.TIMEFRAME_M1):
    # Initialize MT5
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return

    print(f"Connected to MT5. Fetching data for {symbol}...")

    # Get the maximum number of bars allowed by MT5 setting
    # We fetch a very large number, MT5 will return what it has in its buffer
    # depending on "Max bars in chart" setting.
    # Usually users want "as much as possible".
    
    # We'll try to fetch 10 million bars (M1 usually has ~250k bars per year)
    # 10M bars covers ~40 years of 1-minute data if available.
    max_bars = 10_000_000 
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max_bars)

    if rates is None or len(rates) == 0:
        print(f"No rates found for {symbol}. Error: {mt5.last_error()}")
        mt5.shutdown()
        return

    print(f"Fetched {len(rates)} bars.")

    # Create DataFrame
    df = pd.DataFrame(rates)
    
    # Convert time in seconds to datetime objects
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Export to Parquet
    filename = f"{symbol.replace('$', 'S')}_M1_data.parquet"
    output_path = os.path.join(os.getcwd(), filename)
    
    print(f"Exporting to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    print("Export complete.")
    
    # Shut down MT5 connection
    mt5.shutdown()

if __name__ == "__main__":
    export_mt5_data_to_parquet()
