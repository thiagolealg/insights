import json
import requests
from datetime import datetime

def calculate_daily_profit_with_filters():
    # Carregar estratégias salvas
    with open('saved_strategies.json', 'r', encoding='utf-8') as f:
        strategies = json.load(f)
    
    print(f"Total de estratégias salvas: {len(strategies)}")
    
    # Calcular lucro do dia sem filtros (todas as operações)
    print("\n=== LUCRO DO DIA 12/01/2026 ===\n")
    
    # Aplicar filtro de 2026 e pegar resumo
    r = requests.post('http://localhost:5000/api/apply_filters', 
        json={'strategy': 'bull_bear', 'years': [2026]})
    
    r2 = requests.get('http://localhost:5000/api/summary?strategy=bull_bear&filtered=true')
    data = r2.json()
    
    print("SEM FILTROS (todas as operações Alta-Baixa em 2026):")
    print(f"  Total Trades: {data.get('total_trades', 0)}")
    print(f"  Win Rate: {data.get('win_rate', 0):.1f}%")
    print(f"  Lucro Total: {data.get('total_profit', 0):,.0f} pontos")
    print(f"  Lucro Médio: {data.get('avg_profit', 0):.2f} pontos")
    
    # Agora aplicar filtros das estratégias salvas combinadas
    print("\n--- Aplicando filtros das 38 estratégias salvas ---\n")
    
    # A API tem um endpoint para aplicar estratégias combinadas
    # Vamos coletar os filtros relevantes
    bull_bear_strategies = [s for s in strategies if s.get('strategy_type') == 'bull_bear']
    print(f"Estratégias Alta-Baixa salvas: {len(bull_bear_strategies)}")
    
    if bull_bear_strategies:
        # Usar a primeira estratégia como exemplo
        first = bull_bear_strategies[0]
        print(f"\nExemplo - Estratégia: {first.get('name')}")
        print(f"Ratio: {first.get('ratio_label')}")
        filters = first.get('filters', {})
        
        # Adicionar 2026 aos filtros de anos
        years = filters.get('years', [])
        if 2026 not in years:
            years.append(2026)
        filters['years'] = years
        
        # Aplicar filtros
        r3 = requests.post('http://localhost:5000/api/apply_filters', 
            json={'strategy': 'bull_bear', **filters})
        
        r4 = requests.get('http://localhost:5000/api/summary?strategy=bull_bear&filtered=true')
        data2 = r4.json()
        
        print(f"\nCOM FILTROS DA ESTRATÉGIA '{first.get('name')}':")
        print(f"  Total Trades: {data2.get('total_trades', 0)}")
        print(f"  Win Rate: {data2.get('win_rate', 0):.1f}%")
        print(f"  Lucro Total: {data2.get('total_profit', 0):,.0f} pontos")
        print(f"  Lucro Médio: {data2.get('avg_profit', 0):.2f} pontos")

if __name__ == "__main__":
    calculate_daily_profit_with_filters()
