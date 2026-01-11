from strategy_analyzer import StrategyAnalyzer
import pandas as pd
import sys

try:
    # Ajuste o caminho do arquivo de dados conforme necessário (copie do app.py ou use o default do analyzer)
    # Mas como o analyzer precisa do arquivo para gerar o hash do cache, precisamos apontar para o arquivo certo.
    # O arquivo usado no app.py é 'attached_assets/win_1767085916180.txt'
    data_file = 'attached_assets/win_1767085916180.txt'
    analyzer = StrategyAnalyzer(data_file)
    
    if analyzer.load_cache():
        print("Cache carregado com sucesso.")
        if 'slope_degrees' in analyzer.all_trades_df.columns:
            print(analyzer.all_trades_df['slope_degrees'].describe())
            print("\nExemplos (head):")
            print(analyzer.all_trades_df['slope_degrees'].head(20).tolist())
        else:
            print("Coluna 'slope_degrees' não encontrada no cache.")
    else:
        print("Falha ao carregar cache.")
except Exception as e:
    print(f"Erro: {e}")
