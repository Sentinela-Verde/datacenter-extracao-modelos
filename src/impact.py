"""Step 3 — análise temporal e de impacto dos Data Centers.

Este módulo implementa as lógicas extraídas dos notebooks do Colabory para:
- definir os períodos pré, durante e pós;
- calcular métricas por período a partir da saída do Step 2;
- comparar pré vs pós e derivar indicadores de impacto territorial.

As funções aqui são totalmente reutilizáveis em notebooks, scripts e pipelines automatizados.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PERIOD_LABELS = {
    'pre': 'Pré',
    'durante': 'Durante',
    'pos': 'Pós',
}

CLASS_ALIASES = {
    'vegetacao': 'Vegetação',
    'vegetação': 'Vegetação',
    'area construida': 'Construção',
    'área construída': 'Construção',
    'construcao': 'Construção',
    'construção': 'Construção',
    'agua': 'Água',
    'água': 'Água',
    'estrada': 'Estrada',
    'outro': 'Outro',
    'non built': 'Outro',
    'nao construida': 'Outro',
    'não construída': 'Outro',
}


def normalize_class_name(value: str) -> str:
    """Padroniza nomes de classe para o vocabulário do projeto."""
    if value is None or pd.isna(value):
        return ''
    token = str(value).strip().lower()
    for key, mapped in CLASS_ALIASES.items():
        if key == token:
            return mapped
    return str(value).strip()


def define_periods(construction_year: int, operation_year: int | None = None, pre_window: int = 3, post_window: int = 4):
    """Define os períodos de análise temporal a partir do ano de construção e operação.

    Estrutura inspirada no Notebook 04:
    - pré: 3 anos antes da obra
    - durante: ano da obra até 1 ano após a operação
    - pós: após a operação até 4 anos depois
    """
    if operation_year is None:
        operation_year = construction_year

    periodos = {
        'pre': {
            'label': 'Pré',
            'inicio': construction_year - pre_window,
            'fim': construction_year - 1,
        },
        'durante': {
            'label': 'Durante',
            'inicio': construction_year,
            'fim': max(operation_year + 1, construction_year),
        },
        'pos': {
            'label': 'Pós',
            'inicio': operation_year + 1,
            'fim': operation_year + post_window,
        },
    }
    return periodos


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'ano' not in df.columns:
        raise ValueError("O DataFrame deve conter a coluna 'ano'.")
    if 'classe' not in df.columns:
        raise ValueError("O DataFrame deve conter a coluna 'classe'.")
    if 'percentual' not in df.columns:
        raise ValueError("O DataFrame deve conter a coluna 'percentual'.")

    df = df.copy()
    df['classe'] = df['classe'].map(normalize_class_name)
    df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int64')
    df['percentual'] = pd.to_numeric(df['percentual'], errors='coerce').fillna(0)
    if 'area_km2' in df.columns:
        df['area_km2'] = pd.to_numeric(df['area_km2'], errors='coerce').fillna(0)
    return df.dropna(subset=['ano', 'classe'])


def load_processed_table(csv_path: str | Path) -> pd.DataFrame:
    """Carrega a tabela de cobertura por ano produzida pelo Step 2."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    return _ensure_columns(df)


def aggregate_yearly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa os resultados por ano e classe, usando a média entre modelos quando houver mais de um."""
    df = _ensure_columns(df)
    grouped = df.groupby(['ano', 'classe'], as_index=False).agg(
        percentual=('percentual', 'mean'),
        area_km2=('area_km2', 'mean') if 'area_km2' in df.columns else ('percentual', lambda s: 0.0),
    )
    return grouped


def summarize_period(df: pd.DataFrame, periodo: str, construction_year: int, operation_year: int | None = None) -> pd.DataFrame:
    """Calcula métricas agregadas por período (pré, durante, pós)."""
    year_df = aggregate_yearly_metrics(df)
    periods = define_periods(construction_year, operation_year)
    period_cfg = periods[periodo]
    period_df = year_df[(year_df['ano'] >= period_cfg['inicio']) & (year_df['ano'] <= period_cfg['fim'])]

    if period_df.empty:
        return pd.DataFrame([
            {
                'periodo': periodo,
                'label': PERIOD_LABELS.get(periodo, periodo.upper()),
                'ano_inicio': period_cfg['inicio'],
                'ano_fim': period_cfg['fim'],
                'percentual_vegetacao': 0.0,
                'percentual_construcao': 0.0,
                'percentual_agua': 0.0,
                'percentual_estrada': 0.0,
                'percentual_outro': 0.0,
                'area_construida_km2': 0.0,
                'area_vegetacao_km2': 0.0,
                'area_agua_km2': 0.0,
            }
        ])

    pivot = period_df.pivot_table(index='ano', columns='classe', values='percentual', aggfunc='mean').fillna(0)
    pivot_area = period_df.pivot_table(index='ano', columns='classe', values='area_km2', aggfunc='mean').fillna(0)

    row = {
        'periodo': periodo,
        'label': PERIOD_LABELS.get(periodo, periodo.upper()),
        'ano_inicio': period_cfg['inicio'],
        'ano_fim': period_cfg['fim'],
        'percentual_vegetacao': float(pivot.get('Vegetação', 0).mean()) if 'Vegetação' in pivot.columns else 0.0,
        'percentual_construcao': float(pivot.get('Construção', 0).mean()) if 'Construção' in pivot.columns else 0.0,
        'percentual_agua': float(pivot.get('Água', 0).mean()) if 'Água' in pivot.columns else 0.0,
        'percentual_estrada': float(pivot.get('Estrada', 0).mean()) if 'Estrada' in pivot.columns else 0.0,
        'percentual_outro': float(pivot.get('Outro', 0).mean()) if 'Outro' in pivot.columns else 0.0,
        'area_vegetacao_km2': float(pivot_area.get('Vegetação', 0).mean()) if 'Vegetação' in pivot_area.columns else 0.0,
        'area_construida_km2': float(pivot_area.get('Construção', 0).mean()) if 'Construção' in pivot_area.columns else 0.0,
        'area_agua_km2': float(pivot_area.get('Água', 0).mean()) if 'Água' in pivot_area.columns else 0.0,
    }
    return pd.DataFrame([row])


def summarize_temporal_impact(df: pd.DataFrame, construction_year: int, operation_year: int | None = None) -> pd.DataFrame:
    """Cria um resumo completo em três períodos: pré, durante e pós."""
    summaries = []
    for periodo in ['pre', 'durante', 'pos']:
        summaries.append(summarize_period(df, periodo, construction_year, operation_year))
    summary = pd.concat(summaries, ignore_index=True)
    summary['delta_construcao_vs_pre'] = summary['percentual_construcao'] - summary.iloc[0]['percentual_construcao'] if not summary.empty else 0.0
    summary['delta_vegetacao_vs_pre'] = summary['percentual_vegetacao'] - summary.iloc[0]['percentual_vegetacao'] if not summary.empty else 0.0
    summary['delta_agua_vs_pre'] = summary['percentual_agua'] - summary.iloc[0]['percentual_agua'] if not summary.empty else 0.0
    return summary


def compare_pre_vs_post(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a variação entre pré e pós seguindo a lógica do Notebook 04."""
    if summary_df.empty:
        return summary_df
    pre = summary_df[summary_df['periodo'] == 'pre'].iloc[0]
    pos = summary_df[summary_df['periodo'] == 'pos'].iloc[0]

    comparison = pd.DataFrame([
        {
            'periodo_pre': 'pre',
            'periodo_pos': 'pos',
            'delta_area_construida_ha': float(pos['area_construida_km2'] - pre['area_construida_km2']) * 100,
            'delta_vegetacao': float(pos['percentual_vegetacao'] - pre['percentual_vegetacao']),
            'delta_construcao': float(pos['percentual_construcao'] - pre['percentual_construcao']),
            'delta_agua': float(pos['percentual_agua'] - pre['percentual_agua']),
            'delta_estrada': float(pos['percentual_estrada'] - pre['percentual_estrada']),
            'impact_score': (
                float(pos['percentual_construcao'] - pre['percentual_construcao'])
                - float(pos['percentual_vegetacao'] - pre['percentual_vegetacao']) * 0.5
                + float(pos['percentual_agua'] - pre['percentual_agua']) * 0.2
            ),
        }
    ])
    return comparison


def build_impact_report(csv_path: str | Path, facility_name: str, construction_year: int, operation_year: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cria a tabela de resumo temporal e a tabela de comparação pré x pós."""
    df = load_processed_table(csv_path)
    summary = summarize_temporal_impact(df, construction_year, operation_year)
    summary.insert(0, 'facility_name', facility_name)
    delta = compare_pre_vs_post(summary)
    delta.insert(0, 'facility_name', facility_name)
    return summary, delta


def build_impact_summary_file(csv_path: str | Path, facility_name: str, construction_year: int, operation_year: int | None = None, output_dir: str | Path | None = None):
    """Salva um CSV com o resumo dos períodos e uma tabela de deltas pré x pós."""
    summary, delta = build_impact_report(csv_path, facility_name, construction_year, operation_year)
    out_dir = Path(output_dir) if output_dir else Path(csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'{facility_name}_impacto_temporal.csv'
    delta_path = out_dir / f'{facility_name}_delta_pre_pos.csv'
    summary.to_csv(summary_path, index=False)
    delta.to_csv(delta_path, index=False)
    return summary_path, delta_path


def summarize_portfolio_impact(
    processed_dir: str | Path,
    site_years: Dict[str, Tuple[int, int | None] | int | None] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Agrega o impacto de vários data centers em uma tabela única de ranking consolidado.

    Parameters
    ----------
    processed_dir:
        Pasta que contém os CSVs finais do Step 2, como *_cobertura_por_ano.csv.
    site_years:
        Dicionário no formato {nome_do_site: (construction_year, operation_year)}.
        Se omitido, tenta inferir os nomes a partir dos arquivos CSV existentes.
    output_dir:
        Opcional. Diretório para salvar o ranking final em CSV.
    """
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f'Diretório não encontrado: {processed_dir}')

    if site_years is None:
        site_years = {}
        for csv_path in sorted(processed_dir.glob('*_cobertura_por_ano.csv')):
            facility_name = csv_path.stem.replace('_cobertura_por_ano', '')
            site_years[facility_name] = None

    records: List[Dict[str, object]] = []
    for facility_name, year_spec in site_years.items():
        construction_year = None
        operation_year = None

        if year_spec is None:
            csv_candidates = sorted(processed_dir.glob(f'{facility_name}_cobertura_por_ano.csv'))
            if not csv_candidates:
                continue
            construction_year = None
            operation_year = None
        elif isinstance(year_spec, tuple):
            if len(year_spec) == 2:
                construction_year, operation_year = year_spec
            elif len(year_spec) == 1:
                construction_year = year_spec[0]
            else:
                raise ValueError(f'Formato inválido para {facility_name}: {year_spec!r}')
        elif isinstance(year_spec, int):
            construction_year = year_spec
        else:
            raise TypeError(f'Formato inesperado para {facility_name}: {type(year_spec).__name__}')

        csv_candidates = sorted(processed_dir.glob(f'{facility_name}_cobertura_por_ano.csv'))
        if not csv_candidates:
            matched = sorted(processed_dir.glob(f'*{facility_name}*_cobertura_por_ano.csv'))
            if not matched:
                continue
            csv_candidates = matched

        csv_path = csv_candidates[0]
        if construction_year is None and not isinstance(year_spec, tuple):
            # tenta inferir a construção pela presença de dados, quando a informação não foi informada
            construction_year = 2020

        if construction_year is None:
            raise ValueError(
                f'Não foi possível inferir o ano de construção para {facility_name}. '
                'Informe site_years={nome: (ano_construcao, ano_operacao)}.'
            )

        _, delta = build_impact_report(csv_path, facility_name, construction_year, operation_year)
        if delta.empty:
            continue
        row = delta.iloc[0].to_dict()
        records.append(row)

    if not records:
        empty = pd.DataFrame(columns=[
            'facility_name',
            'periodo_pre',
            'periodo_pos',
            'delta_area_construida_ha',
            'delta_vegetacao',
            'delta_construcao',
            'delta_agua',
            'delta_estrada',
            'impact_score',
        ])
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            empty.to_csv(out_dir / 'portfolio_impact_ranking.csv', index=False)
        return empty

    portfolio = pd.DataFrame(records)
    portfolio['impact_score'] = pd.to_numeric(portfolio['impact_score'], errors='coerce').fillna(0.0)
    portfolio['rank'] = portfolio['impact_score'].rank(method='dense', ascending=False).astype(int)
    portfolio = portfolio.sort_values(['impact_score', 'delta_construcao'], ascending=[False, False]).reset_index(drop=True)

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        portfolio.to_csv(out_dir / 'portfolio_impact_ranking.csv', index=False)

    return portfolio


def parse_args():
    parser = argparse.ArgumentParser(description='Análise temporal de impacto ambiental por Data Center.')
    parser.add_argument('--csv', required=True, help='Arquivo CSV produzido pelo Step 2, com colunas ano/classe/percentual')
    parser.add_argument('--name', required=True, help='Nome do Data Center')
    parser.add_argument('--construction-year', type=int, required=True, help='Ano de início da obra')
    parser.add_argument('--operation-year', type=int, default=None, help='Ano de início da operação (opcional)')
    parser.add_argument('--output-dir', default=None, help='Pasta para salvar os CSVs finais')
    return parser.parse_args()


def main():
    args = parse_args()
    summary, delta = build_impact_report(args.csv, args.name, args.construction_year, args.operation_year)
    print('\n' + '=' * 80)
    print(f'IMPACTO TEMPORAL — {args.name}')
    print('=' * 80)
    print(summary.to_string(index=False))
    print('\nVARIAÇÃO PRÉ X PÓS')
    print('=' * 80)
    print(delta.to_string(index=False))

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'{args.name}_impacto_temporal.csv'
    delta_path = out_dir / f'{args.name}_delta_pre_pos.csv'
    summary.to_csv(summary_path, index=False)
    delta.to_csv(delta_path, index=False)
    print(f'\nArquivos salvos em: {summary_path} e {delta_path}')


if __name__ == '__main__':
    main()
