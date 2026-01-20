"""
Análise de Estabilidade das Estratégias Salvas
Verifica quais estratégias são mais consistentes ao longo dos anos.
"""

import json
import requests
import numpy as np
from datetime import datetime

def analyze_strategy_stability():
    with open('saved_strategies.json', 'r', encoding='utf-8') as f:
        strategies = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"  ANÁLISE DE ESTABILIDADE DAS ESTRATÉGIAS (2020-2026)")
    print(f"{'='*80}\n")
    
    resultados = []
    anos_analise = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    
    for strategy in strategies:
        nome = strategy.get('name', 'Sem nome')
        tipo = strategy.get('strategy_type', '?')
        ratio = strategy.get('ratio_label', '?')
        filters = strategy.get('filters', {}).copy()
        
        lucros_por_ano = []
        
        for ano in anos_analise:
            filters_ano = filters.copy()
            filters_ano['years'] = [ano]
            filters_ano['strategy'] = tipo
            
            try:
                r = requests.post('http://localhost:5000/api/apply_filters', json=filters_ano, timeout=10)
                r2 = requests.get(f'http://localhost:5000/api/equity_curve?strategy={tipo}&filtered=true', timeout=10)
                equity = r2.json()
                
                if equity:
                    lucro = float(equity[-1].get('cumulative_result', 0))
                    lucros_por_ano.append(lucro)
                else:
                    lucros_por_ano.append(0)
            except:
                lucros_por_ano.append(0)
        
        # Calcular métricas de estabilidade
        lucros = np.array(lucros_por_ano)
        media = np.mean(lucros)
        std = np.std(lucros)
        
        # Anos positivos
        anos_positivos = sum(1 for l in lucros if l > 0)
        
        # Coeficiente de variação (menor = mais estável)
        cv = abs(std / media) if media != 0 else 999
        
        # Sharpe simplificado (média / desvio)
        sharpe = media / std if std > 0 else 0
        
        # Score de estabilidade (maior = melhor)
        score = (anos_positivos * 10) + (sharpe * 5) - (cv * 2) + (media / 1000)
        
        resultados.append({
            'nome': nome,
            'tipo': tipo,
            'ratio': ratio,
            'media': media,
            'std': std,
            'cv': cv,
            'sharpe': sharpe,
            'anos_positivos': anos_positivos,
            'total': sum(lucros),
            'score': score,
            'lucros': lucros_por_ano
        })
    
    # Ordenar por score de estabilidade
    resultados.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"{'#':<3} {'Nome':<22} {'Tipo':<12} {'Anos+':<6} {'Média':<10} {'CV':<8} {'Score':<8} {'Total':<12}")
    print(f"{'-'*90}")
    
    for i, r in enumerate(resultados[:15], 1):  # Top 15
        media_fmt = f"{r['media']:>+,.0f}" if r['media'] != 0 else "0"
        total_fmt = f"{r['total']:>+,.0f}"
        cv_fmt = f"{r['cv']:.2f}" if r['cv'] < 100 else "Alto"
        print(f"{i:<3} {r['nome'][:22]:<22} {r['tipo'][:12]:<12} {r['anos_positivos']}/7    {media_fmt:<10} {cv_fmt:<8} {r['score']:>6.1f}  {total_fmt}")
    
    print(f"\n{'='*80}")
    print("LEGENDA:")
    print("  Anos+ = Anos com lucro positivo (de 7 analisados: 2020-2026)")
    print("  CV = Coeficiente de Variação (menor = mais estável)")
    print("  Score = Pontuação de estabilidade (maior = melhor)")
    print(f"{'='*80}\n")
    
    # Top 5 mais estáveis
    print("🏆 TOP 5 ESTRATÉGIAS MAIS ESTÁVEIS:")
    for i, r in enumerate(resultados[:5], 1):
        print(f"   {i}. {r['nome']} ({r['tipo']}) - {r['anos_positivos']}/7 anos positivos, Score: {r['score']:.1f}")
    
    return resultados

if __name__ == "__main__":
    analyze_strategy_stability()
