
import pandas as pd
import numpy as np
import os
import sys

def diagnose():
    output_file = 'volatility_diagnosis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        def log(msg):
            print(msg)
            f.write(msg + '\n')

        data_path = 'attached_assets/win_1767085916180.txt'
        if not os.path.exists(data_path):
            log(f"Data file not found: {data_path}")
            return

        log(f"Reading data from {data_path}...")
        try:
            # Ler apenas as colunas necessárias para volatilidade e um número seguro de linhas para ser rápido mas representativo
            df = pd.read_csv(data_path, nrows=500000) 
        except Exception as e:
            log(f"Error reading SCV: {e}")
            return

        columns_needed = ['high', 'low']
        if not all(col in df.columns for col in columns_needed):
            log(f"Missing columns. Found: {df.columns.tolist()}")
            return

        log(f"Data loaded: {len(df)} rows.")

        # Calcular Volatilidade
        log("Calculating volatility (SMA 20 of range)...")
        df['volatility'] = (df['high'] - df['low']).rolling(window=20).mean()

        # Analisar Distribuição
        vol_clean = df['volatility'].dropna()
        log(f"Rows with valid volatility: {len(vol_clean)}")
        
        if len(vol_clean) == 0:
            log("No valid volatility calculated.")
            return

        min_vol = vol_clean.min()
        max_vol = vol_clean.max()
        mean_vol = vol_clean.mean()
        unique_vals = vol_clean.nunique()
        zeros = (vol_clean == 0).sum()

        log(f"Min: {min_vol:.4f}")
        log(f"Max: {max_vol:.4f}")
        log(f"Mean: {mean_vol:.4f}")
        log(f"Unique values: {unique_vals}")
        log(f"Zero values: {zeros}")

        # Testar qcut
        log("\n--- Testing qcut (5 bins, drop duplicates) ---")
        try:
            # Teste padrão que está no código
            out, bins = pd.qcut(vol_clean, q=5, retbins=True, duplicates='drop')
            log(f"Success! Number of bins generated: {len(bins) - 1}")
            log(f"Bins: {bins}")
            
            if len(bins) - 1 < 5:
                log("WARNING: qcut generated fewer than 5 bins due to duplicate edges!")
                log("This will CAUSE ERROR if 5 labels are provided.")
            
            # Testar com Labels
            log("\n--- Testing qcut WITH LABELS ---")
            labels = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
            try:
                out_labels = pd.qcut(vol_clean, q=5, labels=labels, duplicates='drop')
                log("Success with labels!")
                log(f"Counts:\n{out_labels.value_counts()}")
            except ValueError as ve:
                log(f"ValueError with labels: {ve}")
                log("Hypothesis CONFIRMED: Mismatch between bins and labels.")
            except Exception as e:
                log(f"Other error with labels: {e}")

        except Exception as e:
            log(f"Error in qcut logic: {e}")

if __name__ == "__main__":
    diagnose()
