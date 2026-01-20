import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def generate_daily_trades_chart(strategy_type='bull_bear', target_date='2026-01-12'):
    """Gera gráfico e tabela de trades do dia especificado."""
    
    # Carregar dados
    print(f"Carregando dados...")
    df = pd.read_parquet('win_full_data.parquet')
    df['time'] = pd.to_datetime(df['time'])
    
    # Filtrar para o dia
    target = pd.to_datetime(target_date)
    df_day = df[df['time'].dt.date == target.date()].copy()
    print(f"Candles do dia: {len(df_day)}")
    
    # Calcular SMA20
    df['sma20'] = df['close'].rolling(20).mean()
    df_day = df[df['time'].dt.date == target.date()].copy()
    
    # Simular sinais Alta-Baixa (bull_bear) - simplificado
    # Compra: close[i-2] < close[i-1] e preço > SMA20
    # Venda: close[i-2] > close[i-1] e preço < SMA20
    
    signals = []
    close_arr = df_day['close'].values
    time_arr = df_day['time'].values
    sma_arr = df_day['sma20'].values
    
    for i in range(2, len(df_day)):
        # Condição Compra (Alta-Baixa)
        if close_arr[i-2] < close_arr[i-1] and close_arr[i-1] > sma_arr[i-1]:
            signals.append({
                'time': pd.to_datetime(time_arr[i]),
                'price': close_arr[i],
                'direction': 'Compra',
                'color': 'green'
            })
        # Condição Venda (Baixa-Alta)
        elif close_arr[i-2] > close_arr[i-1] and close_arr[i-1] < sma_arr[i-1]:
            signals.append({
                'time': pd.to_datetime(time_arr[i]),
                'price': close_arr[i],
                'direction': 'Venda',
                'color': 'red'
            })
    
    signals_df = pd.DataFrame(signals)
    print(f"Sinais encontrados: {len(signals_df)}")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#0f172a')
    
    # Plot preço
    ax.plot(df_day['time'], df_day['close'], color='#60a5fa', linewidth=1, label='Preço', alpha=0.8)
    ax.plot(df_day['time'], df_day['sma20'], color='#fbbf24', linewidth=1, label='SMA20', alpha=0.6)
    
    # Plot sinais
    if len(signals_df) > 0:
        compras = signals_df[signals_df['direction'] == 'Compra']
        vendas = signals_df[signals_df['direction'] == 'Venda']
        
        ax.scatter(compras['time'], compras['price'], color='#22c55e', marker='^', s=80, label=f'Compra ({len(compras)})', zorder=5)
        ax.scatter(vendas['time'], vendas['price'], color='#ef4444', marker='v', s=80, label=f'Venda ({len(vendas)})', zorder=5)
    
    # Formatação
    ax.set_title(f'Trades Alta-Baixa - {target_date}', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Horário', color='#94a3b8')
    ax.set_ylabel('Preço', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
    ax.grid(True, alpha=0.2, color='#334155')
    
    for spine in ax.spines.values():
        spine.set_color('#334155')
    
    plt.tight_layout()
    
    # Salvar
    output_path = f'trades_{target_date.replace("-", "")}.png'
    plt.savefig(output_path, dpi=150, facecolor='#1e293b', edgecolor='none')
    print(f"Gráfico salvo: {output_path}")
    plt.close()
    
    # Tabela resumo
    if len(signals_df) > 0:
        print("\n=== TABELA DE TRADES DO DIA ===")
        print(f"{'Hora':<10} {'Direção':<10} {'Preço':<10}")
        print("-" * 30)
        for _, row in signals_df.iterrows():
            print(f"{row['time'].strftime('%H:%M'):<10} {row['direction']:<10} {row['price']:<10.0f}")
        print("-" * 30)
        print(f"Total Compras: {len(compras)}")
        print(f"Total Vendas: {len(vendas)}")
        print(f"Total Geral: {len(signals_df)}")
    
    return signals_df

if __name__ == "__main__":
    generate_daily_trades_chart()
