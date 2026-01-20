import json
import requests
from datetime import datetime

def calculate_filtered_today():
    with open('saved_strategies.json', 'r', encoding='utf-8') as f:
        strategies = json.load(f)
    
    total_lucro = 0
    resultados = []
    
    for strategy in strategies:
        nome = strategy.get('name', 'Sem nome')
        tipo = strategy.get('strategy_type', '?')
        ratio = strategy.get('ratio_label', '?')
        filters = strategy.get('filters', {}).copy()
        
        filters['years'] = [2026]
        filters['strategy'] = tipo
        
        try:
            r = requests.post('http://localhost:5000/api/apply_filters', json=filters, timeout=10)
            r2 = requests.get(f'http://localhost:5000/api/equity_curve?strategy={tipo}&filtered=true', timeout=10)
            equity = r2.json()
            
            if equity:
                lucro = float(equity[-1].get('cumulative_result', 0))
                total_lucro += lucro
                resultados.append({'nome': nome, 'lucro': lucro})
        except:
            pass
    
    print(f"TOTAL CONSOLIDADO: {total_lucro:,.0f} pontos")
    return total_lucro

if __name__ == "__main__":
    calculate_filtered_today()
