"""Step 3 — análise temporal e de impacto dos Data Centers.

Executa a análise de impacto a partir do CSV gerado pelo Step 2, comparando os períodos pré,
 durante e pós e produzindo tabelas de variação.

Uso individual:
    python scripts/step3_analise_impacto_temporal.py \
        --csv data/processed/Ascenty_Vinhedo_cobertura_por_ano.csv \
        --name Ascenty_Vinhedo \
        --construction-year 2020 \
        --operation-year 2024

Uso multi-site (ranking consolidado):
    python scripts/step3_analise_impacto_temporal.py \
        --portfolio \
        --processed-dir data/processed \
        --site-years "Ascenty_Vinhedo=2020:2024" \
        --output-dir data/processed
"""
import argparse
from pathlib import Path

from src.impact import build_impact_report, summarize_portfolio_impact


def parse_site_years(raw_value: str):
    mapping = {}
    if not raw_value:
        return mapping
    for item in raw_value.split(';'):
        item = item.strip()
        if not item:
            continue
        name, value = item.split('=', 1)
        name = name.strip()
        value = value.strip()
        if ':' in value:
            construction_year, operation_year = value.split(':', 1)
            mapping[name] = (int(construction_year), int(operation_year) if operation_year else None)
        else:
            mapping[name] = int(value)
    return mapping


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', help='CSV com as séries temporais por classe e ano')
    parser.add_argument('--name', help='Nome do Data Center')
    parser.add_argument('--construction-year', type=int, help='Ano de início da obra')
    parser.add_argument('--operation-year', type=int, default=None, help='Ano de início da operação')
    parser.add_argument('--portfolio', action='store_true', help='Executa o ranking consolidado entre vários sites')
    parser.add_argument('--processed-dir', default=None, help='Pasta com os CSVs gerados pelo Step 2 para análise em portfólio')
    parser.add_argument('--site-years', default=None, help='Mapeamento dos sites: "SITE=ano_construcao:ano_operacao;SITE2=2020"')
    parser.add_argument('--output-dir', default=None, help='Pasta para salvar os CSVs finais')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.portfolio:
        if not args.processed_dir:
            raise ValueError('Para a análise em portfólio, informe --processed-dir.')
        site_years = parse_site_years(args.site_years)
        if not site_years:
            raise ValueError('Para a análise em portfólio, informe --site-years com o formato "NOME=ano_construcao:ano_operacao".')
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.processed_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ranking = summarize_portfolio_impact(args.processed_dir, site_years, output_dir)
        print('\n' + '=' * 80)
        print('RANKING CONSOLIDADO DE IMPACTO - PORTFÓLIO')
        print('=' * 80)
        print(ranking.to_string(index=False))
        print(f'\nArquivo salvo em: {output_dir / "portfolio_impact_ranking.csv"}')
        return

    if not args.csv or not args.name or args.construction_year is None:
        raise ValueError('Para análise individual, informe --csv, --name e --construction-year.')

    summary, delta = build_impact_report(
        args.csv,
        args.name,
        args.construction_year,
        args.operation_year,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f'{args.name}_impacto_temporal.csv'
    delta_path = output_dir / f'{args.name}_delta_pre_pos.csv'

    summary.to_csv(summary_path, index=False)
    delta.to_csv(delta_path, index=False)

    print('\n' + '=' * 80)
    print(f'ANÁLISE TEMPORAL - {args.name}')
    print('=' * 80)
    print(summary.to_string(index=False))
    print('\nVARIAÇÃO PRÉ X PÓS')
    print('=' * 80)
    print(delta.to_string(index=False))
    print(f'\nArquivos salvos em: {summary_path} e {delta_path}')


if __name__ == '__main__':
    main()
