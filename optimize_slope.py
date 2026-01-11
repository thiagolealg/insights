import pandas as pd
import numpy as np
from strategy_analyzer import StrategyAnalyzer

def run_slope_optimization():
    print("Iniciando análise de otimização de inclinação da média...")
    
    # Initialize analyzer and run full backtest
    data_file = "attached_assets/win_1767085916180.txt"
    analyzer = StrategyAnalyzer(data_file)
    analyzer.run_backtest()
    
    trades = analyzer.trades
    if not trades:
        print("Nenhum trade encontrado para análise.")
        return
        
    df_trades = pd.DataFrame(trades)
    
    # Define ranges (bins) for slope in DEGREES (5 in 5 steps)
    bins = list(range(0, 95, 5)) + [1000]
    labels = [f'{i}-{i+5}°' for i in range(0, 90, 5)] + ['90°+']
    
    df_trades['slope_range'] = pd.cut(df_trades['slope_degrees'], bins=bins, labels=labels, right=False)
    
    results = []
    
    for label in labels:
        bin_trades = df_trades[df_trades['slope_range'] == label]
        
        if len(bin_trades) == 0:
            results.append({
                'Ângulo (Graus)': label,
                'Qtd Trades': 0,
                'Win Rate': "0%",
                'Lucro Médio': 0,
                'Profit Factor': 0
            })
            continue
            
        wins = len(bin_trades[bin_trades['winner'] == True])
        win_rate = (wins / len(bin_trades)) * 100
        avg_result = bin_trades['result'].mean()
        
        # Calculate Profit Factor
        gross_profit = bin_trades[bin_trades['result'] > 0]['result'].sum()
        gross_loss = abs(bin_trades[bin_trades['result'] < 0]['result'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        results.append({
            'Ângulo (Graus)': label,
            'Qtd Trades': len(bin_trades),
            'Win Rate': f"{win_rate:.1f}%",
            'Lucro Médio': round(avg_result, 1),
            'Profit Factor': round(profit_factor, 2)
        })
        
    # Display results
    print("\nRelatório de Performance por Ângulo da SMA20 (Teorema de Pitágoras - 5pts/Tick)")
    print("=" * 80)
    report_df = pd.DataFrame(results)
    print(report_df.to_string(index=False))
    print("=" * 80)
    
    # Suggest best slope
    # Criteria: Win Rate > 55% AND Profit Factor > 1.2 AND at least 20 trades
    good_ranges = [r for r in results if r['Qtd Trades'] >= 20 and float(r['Win Rate'].strip('%')) >= 55]
    if good_ranges:
        best = max(good_ranges, key=lambda x: x['Profit Factor'])
        print(f"\nSugestão: A melhor performance está na faixa de {best['Ângulo (Graus)']} de inclinação.")
    else:
        print("\nNenhuma faixa ideal clara encontrada com os critérios mínimos.")

if __name__ == "__main__":
    run_slope_optimization()
