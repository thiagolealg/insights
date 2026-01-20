"""
Calculadora de Lucro em Tempo Real para Estratégias Salvas
Executa a cada minuto quando ativado, ou sob demanda.

Uso:
    python realtime_profit.py          # Executa uma vez
    python realtime_profit.py --loop   # Executa a cada 60 segundos
"""

import json
import requests
import time
from datetime import datetime, date
import sys

BASE_URL = "http://localhost:5000"

def load_saved_strategies():
    """Carrega as estratégias salvas do arquivo JSON."""
    with open('saved_strategies.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_profit_for_strategy(strategy):
    """Calcula o lucro de uma estratégia para o dia de hoje."""
    strategy_type = strategy.get('strategy_type')
    ratio_label = strategy.get('ratio_label')
    filters = strategy.get('filters', {}).copy()
    
    # Garantir que 2026 está nos filtros de ano
    years = filters.get('years', [])
    if 2026 not in years:
        years = years + [2026]
    filters['years'] = years
    filters['strategy'] = strategy_type
    
    try:
        # Aplicar filtros
        r = requests.post(f'{BASE_URL}/api/apply_filters', json=filters, timeout=10)
        if r.status_code != 200:
            return None
        
        # Buscar equity curve para pegar lucro acumulado
        r2 = requests.get(f'{BASE_URL}/api/equity_curve?strategy={strategy_type}&filtered=true', timeout=10)
        equity = r2.json()
        
        if equity:
            ultimo = equity[-1]
            return {
                'lucro': float(ultimo.get('cumulative_result', 0)),
                'ultimo_trade': ultimo.get('entry_time', 'N/A')
            }
        
        return {'lucro': 0, 'ultimo_trade': 'N/A'}
        
    except Exception as e:
        return None

def calculate_all_strategies():
    """Calcula o lucro de todas as estratégias salvas."""
    strategies = load_saved_strategies()
    
    print(f"\n{'='*60}")
    print(f"  LUCRO DAS ESTRATÉGIAS SALVAS - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    total_lucro = 0
    resultados = []
    
    for i, strategy in enumerate(strategies, 1):
        nome = strategy.get('name', 'Sem nome')
        tipo = strategy.get('strategy_type', '?')
        ratio = strategy.get('ratio_label', '?')
        
        result = calculate_profit_for_strategy(strategy)
        
        if result:
            lucro = result['lucro']
            total_lucro += lucro
            
            # Cor baseada no lucro
            cor = "+" if lucro >= 0 else ""
            
            resultados.append({
                'nome': nome,
                'tipo': tipo,
                'ratio': ratio,
                'lucro': lucro
            })
            
            print(f"{i:2}. {nome[:25]:<25} | {tipo[:15]:<15} | {ratio:<15} | {cor}{lucro:>10,.0f} pts")
        else:
            print(f"{i:2}. {nome[:25]:<25} | {tipo[:15]:<15} | ERRO")
    
    print(f"\n{'='*60}")
    print(f"  TOTAL CONSOLIDADO: {total_lucro:>+,.0f} pontos")
    print(f"{'='*60}\n")
    
    # Salvar resultado em arquivo
    with open('ultimo_calculo.txt', 'w', encoding='utf-8') as f:
        f.write(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Total de estratégias: {len(strategies)}\n")
        f.write(f"Lucro consolidado: {total_lucro:,.0f} pontos\n")
    
    return total_lucro, resultados

def main():
    loop_mode = '--loop' in sys.argv
    
    if loop_mode:
        print("Modo loop ativado. Pressione Ctrl+C para parar.")
        while True:
            try:
                calculate_all_strategies()
                print(f"Próxima atualização em 60 segundos...")
                time.sleep(60)
            except KeyboardInterrupt:
                print("\nParado pelo usuário.")
                break
    else:
        calculate_all_strategies()

if __name__ == "__main__":
    main()
