"""
Análise de Estabilidade Otimizada (LOCAL)
Executa a validação de estratégias diretamente em memória, sem chamadas HTTP.
Performance: ~100x mais rápido que a versão via API.
"""

import json
import pandas as pd
import numpy as np
from strategy_analyzer import StrategyAnalyzer
from app import AVAILABLE_STRATEGIES
import time

def analyze_stability_optimized():
    print(f"\n{'='*80}")
    print(f"  ANÁLISE DE ESTABILIDADE OTIMIZADA (2020-2026)")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # 1. Carregar configuração
    try:
        with open('saved_strategies.json', 'r', encoding='utf-8') as f:
            strategies = json.load(f)
    except:
        print("Arquivo saved_strategies.json não encontrado ou inválido.")
        return

    # 2. Carregar dados UMA VEZ na memória
    print("Carregando base de dados (win_full_data.parquet)...")
    try:
        data_file = "win_full_data.parquet"
        # Pré-carregar para evitar reload a cada estratégia
        analyzer_base = StrategyAnalyzer(data_file, 'inside_bar')
        base_df = analyzer_base.all_trades_df.copy() # DF base com indicadores pré-calculados seria ideal, mas aqui pegamos trades
        # Na verdade, StrategyAnalyzer calcula trades no init/run_backtest. 
        # Para otimizar REALMENTE, precisaríamos instanciar o analyzer para cada tipo UMA vez,
        # e depois apenas filtrar o DF de resultados.
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # Agrupar estratégias por tipo para reusar o cálculo do backtest
    strategies_by_type = {}
    for s in strategies:
        stype = s.get('strategy_type')
        if stype not in strategies_by_type:
            strategies_by_type[stype] = []
        strategies_by_type[stype].append(s)
    
    print(f"Estratégias para analisar: {len(strategies)}")
    print(f"Tipos únicos de estratégias: {len(strategies_by_type.keys())}")
    
    results = []
    
    # Processar cada TIPO de estratégia uma vez
    for stype, strat_list in strategies_by_type.items():
        print(f"Processando tipo: {stype} ({len(strat_list)} variantes)...")
        
        # Instanciar e rodar backtest completo para este tipo
        analyzer = StrategyAnalyzer(data_file, stype)
        analyzer.run_backtest()
        
        all_trades = analyzer.all_trades_df
        if all_trades.empty:
            print(f"  Sem trades gerados para {stype}.")
            continue
            
        # Adicionar colunas de ano se não existirem (já devem existir do analyzer)
        if 'year' not in all_trades.columns:
            all_trades['year'] = pd.to_datetime(all_trades['entry_time']).dt.year
            
        # Agora aplicar filtros para cada variante salva
        for strat_config in strat_list:
            nome = strat_config.get('name', 'Sem nome')
            ratio_label = strat_config.get('ratio_label')
            filters = strat_config.get('filters', {})
            
            # Filtrar pelo Ratio
            df_ratio = all_trades[all_trades['ratio_label'] == ratio_label].copy()
            
            if df_ratio.empty:
                continue
                
            lucros_por_ano = []
            anos_analise = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
            
            for ano in anos_analise:
                # Filtrar pelo Ano
                df_ano = df_ratio[df_ratio['year'] == ano].copy()
                
                # Aplicar outros filtros (Hora, Dia da Semana, Volatilidade, DI, Distância SMA)
                # Filter: Hours
                if 'hours' in filters and filters['hours']:
                    df_ano = df_ano[df_ano['hour'].isin(filters['hours'])]
                
                # Filter: Weekdays
                if 'weekdays' in filters and filters['weekdays']:
                    df_ano = df_ano[df_ano['weekday'].isin(filters['weekdays'])]
                
                # Filter: Volatility
                if 'volatility_levels' in filters and filters['volatility_levels']:
                    df_ano = df_ano[df_ano['vol_level'].isin(filters['volatility_levels'])]
                
                # Filter: DI Levels
                if 'di_levels' in filters and filters['di_levels']:
                    df_ano = df_ano[df_ano['di_level'].isin(filters['di_levels'])]
                 
                # Filter: Dist Levels
                if 'dist_levels' in filters and filters['dist_levels']:
                    df_ano = df_ano[df_ano['dist_level'].isin(filters['dist_levels'])]
                
                # Calcular resultado do ano
                if not df_ano.empty:
                    lucro = df_ano['result'].sum()
                else:
                    lucro = 0
                
                lucros_por_ano.append(lucro)
            
            # Calcular estatísticas da variante
            lucros = np.array(lucros_por_ano)
            media = np.mean(lucros)
            std = np.std(lucros)
            anos_positivos = sum(1 for l in lucros if l > 0)
            
            cv = abs(std / media) if media != 0 else 999
            sharpe = media / std if std > 0 else 0
            
            # Score de Estabilidade
            score = (anos_positivos * 10) + (sharpe * 5) - (cv * 2) + (media / 5000)
            
            results.append({
                'nome': nome,
                'tipo': stype,
                'ratio': ratio_label,
                'media': media,
                'std': std,
                'cv': cv,
                'anos_positivos': anos_positivos,
                'total': sum(lucros),
                'score': score,
                'lucros': lucros_por_ano
            })

    # Ordenar e Exibir
    results.sort(key=lambda x: x['score'], reverse=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nANÁLISE CONCLUÍDA EM {duration:.2f} SEGUNDOS")
    print(f"{'='*90}")
    print(f"{'#':<3} {'Nome':<22} {'Tipo':<12} {'Anos+':<6} {'Média':<10} {'CV':<8} {'Score':<8} {'Total':<12}")
    print(f"{'-'*90}")
    
    for i, r in enumerate(results[:20], 1):
        media_fmt = f"{r['media']:>+,.0f}"
        total_fmt = f"{r['total']:>+,.0f}"
        cv_fmt = f"{r['cv']:.2f}" if r['cv'] < 100 else "Alto"
        print(f"{i:<3} {r['nome'][:22]:<22} {r['tipo'][:12]:<12} {r['anos_positivos']}/7    {media_fmt:<10} {cv_fmt:<8} {r['score']:>6.1f}  {total_fmt}")
        
    print(f"{'='*90}")
    
    # Save to file
    with open('stability_report.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    analyze_stability_optimized()
