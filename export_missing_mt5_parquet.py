import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os

def export_filtered_mt5_data(reference_file="attached_assets/win_1767085916180.txt", symbol="WIN$", timeframe=mt5.TIMEFRAME_M1):
    # 1. Get the last date from current data
    try:
        print(f"Reading reference date from {reference_file}...")
        df_ref = pd.read_csv(reference_file)
        df_ref['time'] = pd.to_datetime(df_ref['time'])
        last_date = df_ref['time'].max()
        print(f"Reference last date: {last_date}")
    except Exception as e:
        print(f"Error reading reference: {e}")
        return

    # 2. Init MT5
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return

    print(f"Fetching all available bars for {symbol}...")
    # Fetch 5 million bars (should cover several years)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 5_000_000)
    
    if rates is None or len(rates) == 0:
        print(f"No rates found. Error: {mt5.last_error()}")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 3. Filter data starting AFTER the last date
    df_new = df[df['time'] > last_date].copy()
    
    if df_new.empty:
        print(f"Current MT5 history does not contain any data after {last_date}.")
        mt5.shutdown()
        return

    print(f"Found {len(df_new)} new bars after {last_date}.")
    
    # Sort by time
    df_new.sort_values('time', inplace=True)

    # 4. Save to Parquet
    filename = f"{symbol.replace('$', 'S')}_NEW_M1_data.parquet"
    df_new.to_parquet(filename, index=False)
    
    print(f"File saved: {filename}")
    print(f"Range: {df_new['time'].min()} to {df_new['time'].max()}")
    
    mt5.shutdown()

if __name__ == "__main__":
    export_filtered_mt5_data()
