import json
import requests
from datetime import datetime

def calculate_filtered_today():
    # Carregar estratégias salvas
    with open('saved_strategies.json', 'r', encoding='utf-8') as f:
        strategies = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"  LUCRO DO DIA 12/01/2026 - COM FILTROS DAS ESTRATÉGIAS SALVAS")
    print(f"{'='*70}")
    print(f"{'#':<3} {'Nome':<20} {'Tipo':<15} {'Ratio':<15} {'Lucro':<12}")
    print(f"{'-'*70}")
    
    total_lucro = 0
    count = 0
    
    for i, strategy in enumerate(strategies, 1):
        nome = strategy.get('name', 'Sem nome')[:20]
        tipo = strategy.get('strategy_type', '?')
        ratio = strategy.get('ratio_label', '?')
        filters = strategy.get('filters', {}).copy()
        
        # Garantir 2026 nos anos
        years = filters.get('years', [])
        if 2026 not in years:
            years = years + [2026]
        filters['years'] = [2026]  # Apenas 2026 (hoje)
        filters['strategy'] = tipo
        
        try:
            # Aplicar filtros
            r = requests.post('http://localhost:5000/api/apply_filters', json=filters, timeout=10)
            
            # Buscar equity curve
            r2 = requests.get(f'http://localhost:5000/api/equity_curve?strategy={tipo}&filtered=true', timeout=10)
            equity = r2.json()
            
            if equity:
                lucro = float(equity[-1].get('cumulative_result', 0))
                total_lucro += lucro
                count += 1
                sinal = '+' if lucro >= 0 else ''
                print(f"{i:<3} {nome:<20} {tipo:<15} {ratio:<15} {sinal}{lucro:>10,.0f}")
            else:
                print(f"{i:<3} {nome:<20} {tipo:<15} {ratio:<15} {'N/A':>10}")
        except Exception as e:
            print(f"{i:<3} {nome:<20} ERRO: {str(e)[:30]}")
    
    print(f"{'-'*70}")
    print(f"TOTAL CONSOLIDADO (38 estratégias filtradas): {total_lucro:>+,.0f} pontos")
    print(f"{'='*70}")
    
    return total_lucro

if __name__ == "__main__":
    calculate_filtered_today()
