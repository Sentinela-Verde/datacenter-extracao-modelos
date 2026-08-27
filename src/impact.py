"""Step 3/4 — indicador de impacto (diferença-em-diferenças) e modelo preditivo.

Ainda não implementado — este módulo é só o placeholder previsto na estrutura de pastas de
`modificacoes_projeto.md`, para os próximos passos do projeto:

1. **Painel DiD**: para cada métrica de `data/processed/metricas_painel.csv`, calcular
   `Impacto = (Y_tratado,depois - Y_tratado,antes) - (Y_controle,depois - Y_controle,antes)`,
   usando o ano de chegada/inauguração do data center como corte antes/depois.
2. **Event-study**: plotar cada métrica por "anos relativos à chegada do data center", uma linha
   para tratado e uma para controle — checa a suposição de tendências paralelas no pré-tratamento.
3. **Indicador composto**: combinar via z-score as métricas mais ligadas a impacto físico
   (built-up/IBI, LST, NDVI/EVI) num único índice, mantendo os componentes individuais visíveis.
4. **Modelo preditivo**: `RandomForestRegressor` sobre covariáveis do site (slope, land cover de
   base, distância a vias/área urbana, porte do data center) para prever o indicador de impacto,
   com validação leave-one-out (N pequeno, <=10 sites).
"""
