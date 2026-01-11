
import pandas as pd
import numpy as np
import sys
import os

# Adicionar o diretório atual ao path para importar o StrategyAnalyzer se precisasse, 
# mas vou simular a lógica aqui para ser mais direto e evitar dependências complexas.

def test_volatility_logic():
    print("Testing volatility logic...")
    
    # Simular dados de trades
    # Criar um DataFrame similar ao all_trades_df
    # Vou ler o arquivo de dados original para ter volatilidade real
    
    data_path = 'attached_assets/win_1767085916180.txt'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Reading data from {data_path}")
    try:
        df = pd.read_csv(data_path, nrows=100000)
    except Exception as e:
        print(f"Error reading csv: {e}")
        # Tentar ler com separador diferente se falhar, mas assumindo padrão
        return

    # A coluna de volatilidade deve existir ou ser calculada.
    # No StrategyAnalyzer original:
    # self.data['volatility'] = (self.data['high'] - self.data['low']).rolling(window=20).mean()
    
    if 'high' in df.columns and 'low' in df.columns:
        df['volatility'] = (df['high'] - df['low']).rolling(window=20).mean()
    else:
        print("Columns high/low not found")
        return

    df = df.dropna()
    print(f"Data with volatility: {len(df)} rows")
    
    # Simular all_trades_df duplicando os dados para cada ratio (como no backtest)
    ratios = ['TP 1x : SL 2x', 'TP 2x : SL 1x']
    all_trades = []
    
    for idx, row in df.iterrows():
        # Simplificando, apenas criando linhas com volatilidade
        for r in ratios:
            all_trades.append({
                'entry_index': idx,
                'volatility': row['volatility'],
                'ratio_label': r,
                'result': np.random.randn(), # Random result
                'winner': np.random.choice([True, False])
            })
            
    all_trades_df = pd.DataFrame(all_trades)
    print(f"Created all_trades_df with {len(all_trades_df)} rows")
    
    # APLICAR A LÓGICA DO FIX
    print("Calculating vol_level...")
    try:
        all_trades_df['vol_level'] = pd.qcut(
            all_trades_df['volatility'],
            q=5,
            labels=['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta'],
            duplicates='drop'
        ).astype(str)
        print("Success!")
        print(all_trades_df['vol_level'].value_counts())
        print("Sample vol_level values:", all_trades_df['vol_level'].head().tolist())
        print("Dtypes:", all_trades_df.dtypes)
    except Exception as e:
        print(f"Error in qcut: {e}")

    # TESTAR FILTRO
    print("\nTesting filter...")
    ratio_label = 'TP 1x : SL 2x'
    trades_df = all_trades_df[all_trades_df['ratio_label'] == ratio_label].copy()
    print(f"Trades for ratio {ratio_label}: {len(trades_df)}")
    
    if 'vol_level' in trades_df.columns:
        print("vol_level column exists in trades_df")
    else:
        print("vol_level column MISSING in trades_df")
        
    # Filtrar
    filter_list = ['Alta', 'Muito Alta']
    filtered = trades_df[trades_df['vol_level'].isin(filter_list)]
    print(f"Filtered (Alta, Muito Alta): {len(filtered)}")
    
    # Testar get_stats
    print("\nTesting stats generation...")
    stats = []
    for vol_level in ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']:
        vol_trades = trades_df[trades_df['vol_level'].astype(str) == vol_level]
        print(f"Level {vol_level}: {len(vol_trades)} trades")

if __name__ == "__main__":
    test_volatility_logic()
