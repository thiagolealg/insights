# Plano de Implementação - Atualização de Filtros Take/Stop com Proporção RR

O objetivo é incluir a proporção de Take x Stop (Ratio de Risco/Retorno) nos rótulos dos filtros e estatísticas, facilitando a análise de performance baseada no perfil de cada trade.

## Mudanças Propostas

### 1. `strategy_analyzer.py`
- **Lógica de Categorização**:
    - Continuar dividindo os trades em 5 quintis baseados no tamanho do Stop Loss (`sl_points`).
    - Para cada quintil, calcular a média do `rr_ratio` (Take / Stop).
    - Gerar labels dinâmicos como: `"Muito Curto (1:0.45)"`, `"Longo (1:0.55)"`, etc.
- **Novas Regras de Estratégia**:
  - [x] **Comprehensive "Distance from i-2 to SMA" Filter Feature**
  - [x] **Backend**: Calculate `dist_sma_i2` (abs diff close[i-2] - sma20[i-2]).
  - [x] **Backend**: Categorize into 10 deciles (D1-D10) using `pd.qcut`.
  - [x] **Backend**: Add `dist_levels` to filter logic (`apply_filters`).
  - [x] **Backend**: Create API endpoint `/api/by_dist_sma`.
  - [x] **Frontend**: Add interactive filter group (checkboxes D1-D10).
  - [x] **Frontend**: Add Bar Chart for Win Rate/Profit by Decile.
  - [x] **Frontend**: Add Details Table for stats by Decile.
  - [x] **Frontend**: Update Strategy Persistence (Save/Load/Update) to handle `dist_levels`.
  - [x] **Verification**: Test filters work and chart updates correctly.
  - [x] **Micro Channel Strategy Implementation**
  - [x] **Backend**: Update `StrategyAnalyzer` to support `strategy_type` ('inside_bar', 'micro_channel').
  - [x] **Backend**: Implement `check_buy_conditions_micro` (3 Lower Lows) and `check_sell_conditions_micro` (3 Higher Highs).
  - [x] **Backend**: Adjust `simulate_trade` logic for Micro Channel (range/stop based on i-1).
  - [x] **API**: Update `app.py` to instantiate and route requests to multiple analyzers.
  - [x] **Frontend**: Add Tab Navigation (Inside Bar / Micro Channel).
  - [x] **Frontend**: Update JS to handle strategy state (`currentStrategy`) and pass to API.
  - [x] **Frontend**: Dynamic Rules Display based on active tab.
- [x] **Performance & UX Improvements**
  - [x] **Backend**: Implement Disk Caching (`.pkl`) for backtest results to enable instant startup.
  - [x] **Backend**: Fix race conditions in cache directory creation.
  - [x] **Frontend**: Add Status Badges (🟢/🟡/🔴) to tabs with polling logic.
- **Métodos a Atualizar**:
    - `check_buy_conditions` / `check_sell_conditions`: Para incluir o candle `i-13`.
    - `get_available_filters`: Para retornar os novos nomes dos filtros.
    - `get_stats_by_take_stop`: Para usar os nomes com proporção.
    - `apply_filters`: Para filtrar corretamente usando os novos nomes.

### 2. `templates/index.html`
- Nenhuma mudança é necessária no frontend, pois ele já consome os labels dinamicamente do backend através da API `/api/available_filters`.

## Passos de Verificação

1. **Reiniciar o Servidor**: Garantir que as novas lógicas de cálculo sejam aplicadas.
2. **Navegar no Browser**:
    - Verificar se os filtros na seção "Tamanho do Stop" agora mostram a proporção (ex: "SL Curto (1:0.48)").
    - Verificar se o gráfico e a tabela de estatísticas refletem esses novos nomes.
    - Testar a aplicação de um filtro para confirmar se a integração continua funcionando.
