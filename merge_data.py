import pandas as pd
import os

def merge_and_save_data(old_txt="attached_assets/win_1767085916180.txt", new_parquet="WINS_NEW_M1_data.parquet", output_parquet="win_full_data.parquet"):
    print(f"Loading old data from {old_txt}...")
    df_old = pd.read_csv(old_txt)
    df_old['time'] = pd.to_datetime(df_old['time'])
    
    print(f"Loading new data from {new_parquet}...")
    df_new = pd.read_parquet(new_parquet)
    # Ensure columns match
    # Old columns: time,open,high,low,close,tick_volume,spread,real_volume
    # Parquet usually has the same if exported from my script
    
    print("Merging dataframes...")
    df_full = pd.concat([df_old, df_new]).drop_duplicates(subset=['time']).sort_values('time')
    
    print(f"Merged rows: {len(df_full)}")
    print(f"New range: {df_full['time'].min()} to {df_full['time'].max()}")
    
    print(f"Saving merged data to {output_parquet}...")
    df_full.to_parquet(output_parquet, index=False)
    print("Merge complete.")

if __name__ == "__main__":
    merge_and_save_data()
