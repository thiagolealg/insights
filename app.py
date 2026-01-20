import os
import threading
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime
from strategy_analyzer import StrategyAnalyzer

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Dicionário para armazenar as instâncias das estratégias
analyzers = {}
analyzer_lock = threading.Lock()

# Define as estratégias disponíveis e seus rótulos
AVAILABLE_STRATEGIES = {
    'inside_bar': 'Inside Bar',
    'micro_channel': 'Micro Canal',
    'bull_bear': 'Alta-Baixa (PB Simples)',
    'bull_bear_bear': 'Alta-Baixa-Baixa (PB Duplo)',
    'sequence_reversal': 'Sequência Reversão (6 Bulls 3 Bears)',
    'sma_trend': 'Tendência SMA (Venda)',
    'std_reversal': 'Reversão Desvio Padrão (Nova)',
    'sma_pullback': 'Pullback SMA (i-5)',
    'three_soldiers': 'Três Soldados (Alta-Baixa)',
    'breakout_momentum': 'Breakout Momentum (Corpo 3x)'
}

# Inicializa o status de todas as estratégias como 'loading'
analyzers_status = {strategy_type: 'loading' for strategy_type in AVAILABLE_STRATEGIES.keys()}
analyzers_errors = {strategy_type: None for strategy_type in AVAILABLE_STRATEGIES.keys()}

DATA_FILE = "win_full_data.parquet"

def load_strategy_instance(strategy_type):
    global analyzers, analyzers_status, analyzers_errors
    try:
        print(f"Starting initialization for strategy: {strategy_type}...")
        analyzer = StrategyAnalyzer(DATA_FILE, strategy_type=strategy_type)
        print(f"Running backtest for {strategy_type}...")
        analyzer.run_backtest()
        
        with analyzer_lock:
            analyzers[strategy_type] = analyzer
            analyzers_status[strategy_type] = 'ready'
            
        print(f"Strategy {strategy_type} ready. Trades: {len(analyzer.trades)}")
    except Exception as e:
        analyzers_errors[strategy_type] = str(e)
        analyzers_status[strategy_type] = 'error'
        print(f"Error initializing strategy {strategy_type}: {e}")
        import traceback
        traceback.print_exc()

def get_active_analyzer_instance():
    """Retorna a instância do analyzer baseada no parâmetro 'strategy' da requisição"""
    # Padrão: inside_bar
    strategy_type = request.args.get('strategy')
    
    # Se não estiver no args, tenta no JSON (para POST)
    if not strategy_type and request.is_json:
        data = request.get_json()
        if data:
            strategy_type = data.get('strategy')
    
    if not strategy_type:
        strategy_type = 'inside_bar'
        
    global analyzers
    with analyzer_lock:
        return analyzers.get(strategy_type)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/api/status')
def get_status():
    global analyzers_status, analyzers_errors
    # Retorna status geral. Se algum estiver pronto, o front pode começar a carregar algo?
    # Melhor retornar o status de ambos.
    return jsonify({
        'strategies': analyzers_status,
        'errors': analyzers_errors
    })

@app.route('/api/summary')
def get_summary():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_summary(use_filtered))

@app.route('/api/by_hour')
def get_by_hour():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_hour(use_filtered))

@app.route('/api/by_weekday')
def get_by_weekday():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_weekday(use_filtered))

@app.route('/api/by_year')
def get_by_year():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_year(use_filtered))

@app.route('/api/by_month')
def get_by_month():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_month(use_filtered))

@app.route('/api/by_volatility')
def get_by_volatility():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_volatility(use_filtered))

@app.route('/api/by_direction')
def get_by_direction():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_direction(use_filtered))

@app.route('/api/by_angle')
def get_by_angle():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_angle(use_filtered))

@app.route('/api/by_dist_sma')
def get_by_dist_sma():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_dist_sma(use_filtered))

@app.route('/api/by_di')
def get_by_di():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_di(use_filtered))

@app.route('/api/by_acc')
def get_by_acc():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_acc(use_filtered))

@app.route('/api/by_vol_slope')
def get_by_vol_slope():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_vol_slope(use_filtered))

@app.route('/api/by_jerk')
def get_by_jerk():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_jerk(use_filtered))

@app.route('/api/by_take_stop')
def get_by_take_stop():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_stats_by_take_stop(use_filtered))

@app.route('/api/trades')
def get_trades():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    trades = a.get_recent_trades(100, use_filtered)
    return jsonify(trades)

@app.route('/api/equity_curve')
def get_equity_curve():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    use_filtered = request.args.get('filtered', 'false').lower() == 'true'
    return jsonify(a.get_equity_curve(use_filtered))

@app.route('/api/available_filters')
def get_available_filters():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    return jsonify(a.get_available_filters())

@app.route('/api/apply_filters', methods=['POST'])
def apply_filters():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    filters = request.get_json() or {}
    # Separate strategy param from actual filters if mixed?
    # request.get_json() contains everything. 'apply_filters' method expects filter dict.
    # It will ignore extra keys ideally.
    a.apply_filters(filters)
    return jsonify({'status': 'ok', 'trades_count': len(a.filtered_df)})

@app.route('/api/reset_filters', methods=['POST'])
def reset_filters():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    a.reset_filters()
    return jsonify({'status': 'ok', 'trades_count': len(a.filtered_df)})

@app.route('/api/set_ratio', methods=['POST'])
def set_ratio():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    data = request.get_json() or {}
    ratio_label = data.get('ratio_label', 'TP 1x : SL 2x')
    
    trades_count = a.set_active_ratio(ratio_label)
    return jsonify({
        'status': 'ok', 
        'ratio_label': ratio_label,
        'trades_count': trades_count
    })

@app.route('/api/current_ratio')
def get_current_ratio():
    a = get_active_analyzer_instance()
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    return jsonify({
        'ratio_label': a.current_ratio,
        'available_ratios': [r['label'] for r in a.AVAILABLE_RATIOS]
    })

def init_analyzer():
    # Iniciar threads separadas para não bloquear o servidor
    for strategy_key in AVAILABLE_STRATEGIES.keys():
        t = threading.Thread(target=load_strategy_instance, args=(strategy_key,))
        t.start()

@app.route('/api/debug_data')
def debug_data():
    a = get_active_analyzer_instance()
    if not a or not hasattr(a, 'trades_df'):
        return jsonify({'error': 'Analyzer not ready'})
    
    try:
        df = a.trades_df
        if len(df) == 0:
             return jsonify({'status': 'empty'})
        
        info = {
            'columns': df.columns.tolist(),
            'total_rows': len(df),
            'vol_level_exists': 'vol_level' in df.columns,
            'vol_level_unique': df['vol_level'].unique().tolist() if 'vol_level' in df.columns else [],
            'sample_data': df[['entry_time', 'vol_level']].head(5).to_dict('records') if 'vol_level' in df.columns else [],
            'dtypes': {k: str(v) for k, v in df.dtypes.items()}
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

# ============== STRATEGY MANAGEMENT ==============
import json
import uuid
from flask import Response

STRATEGIES_FILE = 'saved_strategies.json'
STRATEGIES_CACHE = None
STRATEGIES_CACHE_MTIME = 0

def load_strategies():
    """Load saved strategies from file with caching"""
    global STRATEGIES_CACHE, STRATEGIES_CACHE_MTIME
    
    if os.path.exists(STRATEGIES_FILE):
        try:
            mtime = os.path.getmtime(STRATEGIES_FILE)
            if STRATEGIES_CACHE is None or mtime > STRATEGIES_CACHE_MTIME:
                print(f"Loading strategies from disk (mtime: {mtime})...")
                with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
                    STRATEGIES_CACHE = json.load(f)
                STRATEGIES_CACHE_MTIME = mtime
            return STRATEGIES_CACHE
        except Exception as e:
            print(f"Error loading strategies: {e}")
            return []
    return []

def save_strategies(strategies):
    """Save strategies to file and update cache"""
    global STRATEGIES_CACHE, STRATEGIES_CACHE_MTIME
    try:
        with open(STRATEGIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
        
        # Update cache immediately
        STRATEGIES_CACHE = strategies
        STRATEGIES_CACHE_MTIME = os.path.getmtime(STRATEGIES_FILE)
    except Exception as e:
        print(f"Error saving strategies: {e}")

@app.route('/api/strategies', methods=['GET'])
def list_strategies():
    """List saved strategies filtered by current strategy type"""
    target_strategy = request.args.get('strategy')
    strategies = load_strategies()
    
    filtered_strategies = []
    for s in strategies:
        # Legacy strategies (no type) are assumed to be 'inside_bar'
        s_type = s.get('strategy_type', 'inside_bar')
        
        # If no target specified, show all? Or default to inside_bar?
        # User wants separation. Let's filter strictly if param provided.
        if target_strategy and target_strategy != s_type:
            continue
            
        filtered_strategies.append({
            'id': s['id'],
            'name': s['name'],
            'created_at': s['created_at'],
            'description': s.get('description', ''),
            'ratio_label': s.get('ratio_label', 'N/A'),
            'strategy_type': s_type
        })
        
    # Sort by creation date desc
    filtered_strategies.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(filtered_strategies)

@app.route('/api/strategies', methods=['POST'])
def save_strategy():
    """Save current filter configuration as a new strategy"""
    data = request.get_json() or {}
    name = data.get('name', 'Estratégia sem nome')
    description = data.get('description', '')
    filters = data.get('filters', {})
    ratio_label = data.get('ratio_label', 'TP 1x : SL 2x')
    strategy_type = data.get('strategy_type', 'inside_bar')
    
    strategies = load_strategies()
    
    new_strategy = {
        'id': str(uuid.uuid4()),
        'name': name,
        'description': description,
        'created_at': datetime.now().isoformat(),
        'filters': filters,
        'ratio_label': ratio_label,
        'strategy_type': strategy_type
    }
    
    strategies.append(new_strategy)
    save_strategies(strategies)
    
    return jsonify({'status': 'ok', 'strategy': new_strategy})

@app.route('/api/strategies/<strategy_id>', methods=['PUT'])
def update_strategy(strategy_id):
    """Update an existing strategy with current data"""
    data = request.get_json() or {}
    
    strategies = load_strategies()
    updated = False
    
    for s in strategies:
        if s['id'] == strategy_id:
            if 'name' in data: s['name'] = data['name']
            if 'description' in data: s['description'] = data['description']
            if 'filters' in data: s['filters'] = data['filters']
            if 'ratio_label' in data: s['ratio_label'] = data['ratio_label']
            updated = True
            break
            
    if updated:
        save_strategies(strategies)
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>', methods=['GET'])
def get_strategy(strategy_id):
    """Get a specific strategy by ID (for loading)"""
    strategies = load_strategies()
    for s in strategies:
        if s['id'] == strategy_id:
            return jsonify(s)
    return jsonify({'error': 'Strategy not found'}), 404

@app.route('/api/strategies/<strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id):
    """Delete a strategy by ID"""
    strategies = load_strategies()
    strategies = [s for s in strategies if s['id'] != strategy_id]
    save_strategies(strategies)
    return jsonify({'status': 'ok'})

@app.route('/api/strategies/<strategy_id>/export', methods=['GET'])
def export_strategy(strategy_id):
    """Export a strategy as downloadable JSON file"""
    strategies = load_strategies()
    for s in strategies:
        if s['id'] == strategy_id:
            response = Response(
                json.dumps(s, ensure_ascii=False, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment;filename={s["name"]}.json'}
            )
            return response
    return jsonify({'error': 'Strategy not found'}), 404

@app.route('/api/strategies/import', methods=['POST'])
def import_strategy():
    """Import a strategy from uploaded JSON"""
    if 'file' in request.files:
        file = request.files['file']
        try:
            content = json.load(file)
            content['id'] = str(uuid.uuid4())
            content['created_at'] = datetime.now().isoformat()
            content['name'] = content.get('name', 'Estratégia Importada')
            
            strategies = load_strategies()
            strategies.append(content)
            save_strategies(strategies)
            
            return jsonify({'status': 'ok', 'strategy': content})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return jsonify({'error': 'No file provided'}), 400

@app.route('/api/all_strategies_stats', methods=['GET'])
def get_all_strategies_stats():
    """Retorna métricas de todas as estratégias salvas"""
    strategies = load_strategies()
    results = []
    all_trades_list = []
    
    for s in strategies:
        s_type = s.get('strategy_type', 'inside_bar')
        ratio_label = s.get('ratio_label', 'TP 1x : SL 2x')
        filters = s.get('filters', {})
        
        # Pegar o analyzer correspondente ao tipo da estratégia
        with analyzer_lock:
            analyzer = analyzers.get(s_type)
        
        if not analyzer:
            results.append({
                'id': s['id'],
                'name': s['name'],
                'strategy_type': s_type,
                'ratio_label': ratio_label,
                'status': 'loading',
                'error': f'Analyzer {s_type} ainda não está pronto'
            })
            continue
        
        try:
            # Aplica os filtros específicos da estratégia e obtém os trades
            df = analyzer.get_result_for_filters(ratio_label, filters)
            
            if df.empty:
                results.append({
                    'id': s['id'],
                    'name': s['name'],
                    'strategy_type': s_type,
                    'ratio_label': ratio_label,
                    'trades': 0,
                    'win_rate': 0,
                    'sharpe': 0,
                    'profit_factor': 0,
                    'total_profit': 0,
                    'avg_profit': 0
                })
                continue
            
            # Adiciona trades à lista consolidadas para stats anuais
            if 'year' not in df.columns and 'exit_time' in df.columns:
                 df['year'] = pd.to_datetime(df['exit_time']).dt.year
            
            if 'year' in df.columns:
                 # Adiciona apenas colunas necessárias para economizar memória e evitar warnings
                 # Incluindo 'exit_time', 'direction' e injetando 'strategy_name' para trades recentes
                 # Também incluindo 'dist_level' e 'di_level' para stats detalhadas
                 df['strategy_name'] = s['name']
                 cols = ['year', 'result', 'winner', 'exit_time', 'direction', 'strategy_name', 'dist_level', 'di_level']
                 # Ensure all cols exist
                 valid_cols = [c for c in cols if c in df.columns]
                 all_trades_list.append(df[valid_cols].copy())

            # Calcula métricas da estratégia
            total = len(df)
            winners = df['winner'].sum()
            win_rate = (winners / total) * 100 if total > 0 else 0
            total_profit = df['result'].sum()
            avg_profit = df['result'].mean()
            
            # Sharpe ratio
            returns = df['result']
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Profit Factor
            gross_profit = df[df['winner']]['result'].sum()
            gross_loss = abs(df[~df['winner']]['result'].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
            
            results.append({
                'id': s['id'],
                'name': s['name'],
                'strategy_type': s_type,
                'ratio_label': ratio_label,
                'trades': int(total),
                'win_rate': round(float(win_rate), 2),
                'sharpe': round(float(sharpe), 2),
                'profit_factor': round(float(profit_factor), 2),
                'total_profit': round(float(total_profit), 2),
                'avg_profit': round(float(avg_profit), 2)
            })
        except Exception as e:
            results.append({
                'id': s['id'],
                'name': s['name'],
                'strategy_type': s_type,
                'ratio_label': ratio_label,
                'status': 'error',
                'error': str(e)
            })
    
    # Ordenar resultados por profit factor descrescente
    results.sort(key=lambda x: x.get('profit_factor', 0), reverse=True)
    
    # Calcular stats anuais consolidadas
    yearly_stats = []
    if all_trades_list:
        try:
            combined_df = pd.concat(all_trades_list)
            if 'year' in combined_df.columns:
                grouped = combined_df.groupby('year')
                for year, group in grouped:
                    total = len(group)
                    winners = group['winner'].sum()
                    win_rate = (winners / total) * 100 if total > 0 else 0
                    total_profit = group['result'].sum()
                    
                    returns = group['result']
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                    
                    gross_profit = group[group['winner']]['result'].sum()
                    gross_loss = abs(group[~group['winner']]['result'].sum())
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
                    
                    yearly_stats.append({
                        'year': int(year),
                        'trades': int(total),
                        'win_rate': round(float(win_rate), 2),
                        'sharpe': round(float(sharpe), 2),
                        'profit_factor': round(float(profit_factor), 2),
                        'total_profit': round(float(total_profit), 2)
                    })
                
                # Ordenar por ano (mais recente primeiro)
                yearly_stats.sort(key=lambda x: x['year'], reverse=True)
        except Exception as e:
            print(f"Erro calculando stats anuais: {e}")

    # Calcular trades recentes consolidados (Top 100) e outras estatísticas detalhadas
    recent_trades = []
    dist_stats = []
    di_stats = []
    monthly_stats = []

    if all_trades_list:
        try:
            combined_df = pd.concat(all_trades_list)
            
            # --- Recent Trades ---
            if 'exit_time' in combined_df.columns:
                combined_df['exit_time'] = pd.to_datetime(combined_df['exit_time'])
                
                # Clone para recent trades sorting
                df_recent = combined_df.copy()
                df_recent.sort_values('exit_time', ascending=False, inplace=True)
                top_trades = df_recent.head(100)
                for _, row in top_trades.iterrows():
                     recent_trades.append({
                        'strategy': row.get('strategy_name', 'Unknown'),
                        'exit_time': row['exit_time'].strftime('%d/%m/%Y %H:%M'),
                        'result': float(row['result']),
                        'winner': bool(row['winner']),
                        'direction': row.get('direction', '-')
                     })

                # --- Monthly Stats ---
                combined_df['month_key'] = combined_df['exit_time'].dt.strftime('%Y-%m')
                grouped_month = combined_df.groupby('month_key')
                for month, group in grouped_month:
                    total = len(group)
                    winners = group['winner'].sum()
                    win_rate = (winners / total) * 100 if total > 0 else 0
                    total_profit = group['result'].sum()
                    returns = group['result']
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                    monthly_stats.append({
                        'month': month,
                        'trades': int(total),
                        'win_rate': round(float(win_rate), 2),
                        'sharpe': round(float(sharpe), 2),
                        'total_profit': round(float(total_profit), 2),
                        'avg_profit': round(float(total_profit / total), 2) if total > 0 else 0
                    })
                monthly_stats.sort(key=lambda x: x['month'], reverse=True)

            # --- Dist SMA Stats ---
            if 'dist_level' in combined_df.columns:
                grouped_dist = combined_df.groupby('dist_level')
                for dist, group in grouped_dist:
                    total = len(group)
                    winners = group['winner'].sum()
                    win_rate = (winners / total) * 100 if total > 0 else 0
                    total_profit = group['result'].sum()
                    returns = group['result']
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                    dist_stats.append({
                        'dist_level': dist,
                        'trades': int(total),
                        'win_rate': round(float(win_rate), 2),
                        'sharpe': round(float(sharpe), 2),
                        'total_profit': round(float(total_profit), 2),
                        'avg_profit': round(float(total_profit / total), 2) if total > 0 else 0
                    })
                # Sort D1..D10 logic if needed, simple string sort works for D1, D10, D2... (needs natural sort)
                # D1, D2, ..., D9, D10. String sort gives D1, D10, D2. 
                # Custom sort:
                def dist_sort_key(k):
                    if k.startswith('D'):
                        try: return int(k[1:])
                        except: return 999
                    return 999
                dist_stats.sort(key=lambda x: dist_sort_key(x['dist_level']))

            # --- DI Stats ---
            if 'di_level' in combined_df.columns:
                grouped_di = combined_df.groupby('di_level')
                for di, group in grouped_di:
                    total = len(group)
                    winners = group['winner'].sum()
                    win_rate = (winners / total) * 100 if total > 0 else 0
                    total_profit = group['result'].sum()
                    returns = group['result']
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                    di_stats.append({
                        'di_level': di,
                        'trades': int(total),
                        'win_rate': round(float(win_rate), 2),
                        'sharpe': round(float(sharpe), 2),
                        'total_profit': round(float(total_profit), 2),
                        'avg_profit': round(float(total_profit / total), 2) if total > 0 else 0
                    })
                def di_sort_key(k):
                    if k.startswith('DI'):
                        try: return int(k[2:])
                        except: return 999
                    return 999
                di_stats.sort(key=lambda x: di_sort_key(x['di_level']))

        except Exception as e:
            print(f"Erro calculando stats detalhadas: {e}")
            
    return jsonify({
        'strategies': results,
        'yearly_stats': yearly_stats,
        'recent_trades': recent_trades,
        'dist_stats': dist_stats,
        'di_stats': di_stats,
        'monthly_stats': monthly_stats
    })

@app.route('/api/strategies/apply_combined', methods=['POST'])
def apply_combined_strategies():
    """Apply multiple strategies combined into the analysis state"""
    data = request.get_json() or {}
    strategy_ids = data.get('ids', [])
    
    # ATENÇÃO: apply_combined precisa saber qual BASE (analyzer) usar. 
    # Assumindo inside_bar por enquanto para salvar estratégias, mas idealmente estratégias poderiam ser salvas por tipo.
    # Como as estratégias salvas são apenas Filtros JSON, elas podem teoricamente ser aplicadas em qualquer base,
    # mas os resultados serão diferentes.
    # Vou usar get_active_analyzer() aqui também, então o frontend decide onde aplicar.
    
    if not strategy_ids:
        return jsonify({'error': 'Nenhuma estratégia selecionada'}), 400
        
    strategies = load_strategies()
    selected_data = [s for s in strategies if s['id'] in strategy_ids]
    
    if not selected_data:
        return jsonify({'error': 'Estratégias não encontradas'}), 404
        
    a = get_active_analyzer_instance() # Pega o analyzer ativo (inside ou micro)
    if not a: return jsonify({'error': 'Strategy not ready'}), 503
    
    with analyzer_lock:
        count = a.set_combined_filtered_df(selected_data)
        
    return jsonify({'status': 'ok', 'count': count})

if __name__ == '__main__':
    init_analyzer()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

