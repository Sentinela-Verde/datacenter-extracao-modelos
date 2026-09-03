"""Script de teste do pipeline com múltiplos data centers.

Permite:
1. Listar todos os data centers disponíveis
2. Selecionar um subset para testes
3. Executar Step 1, Step 2, Step 3 automaticamente
4. Gerar relatório consolidado de impacto

Uso:
    python scripts/test_pipeline_multi_site.py --list                    # Lista todos os DCs
    python scripts/test_pipeline_multi_site.py --test-small              # Testa 3 DCs (quick test)
    python scripts/test_pipeline_multi_site.py --test-medium             # Testa 8 DCs (medium test)
    python scripts/test_pipeline_multi_site.py --test-full               # Testa todos (full test)
    python scripts/test_pipeline_multi_site.py --test-operators Ascenty  # Apenas Ascenty
    python scripts/test_pipeline_multi_site.py --test-state SP           # Apenas SP
    python scripts/test_pipeline_multi_site.py --test-ids 1,3,5,7        # IDs específicos
"""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Caminho do arquivo de referência
REFERENCE_FILE = Path('data_centers_referencia.json')
SKIP_EE = True  # Se True, não executa Step 1 (evita custo de downloads)


def load_data_centers() -> Dict:
    """Carrega a lista de data centers do arquivo JSON."""
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {REFERENCE_FILE}")
    with open(REFERENCE_FILE) as f:
        return json.load(f)


def print_data_centers(data: Dict):
    """Exibe todos os data centers em formato tabular."""
    df = pd.DataFrame(data['data_centers'])
    print("\n" + "="*120)
    print("LISTA DE DATA CENTERS DISPONÍVEIS")
    print("="*120)
    print(df[['facility_id', 'name', 'operator', 'state', 'city', 'confidence']].to_string(index=False))
    print(f"\nTotal: {len(df)} data centers")
    print("="*120 + "\n")


def filter_data_centers(data: Dict, filter_type: str, filter_value: str) -> List[Dict]:
    """Filtra data centers por tipo."""
    all_dcs = data['data_centers']
    
    if filter_type == 'operator':
        return [dc for dc in all_dcs if dc['operator'].lower() == filter_value.lower()]
    elif filter_type == 'state':
        return [dc for dc in all_dcs if dc['state'].lower() == filter_value.lower()]
    elif filter_type == 'ids':
        ids = [int(x.strip()) for x in filter_value.split(',')]
        return [dc for dc in all_dcs if dc['facility_id'] in ids]
    elif filter_type == 'confidence':
        return [dc for dc in all_dcs if dc['confidence'].lower() == filter_value.lower()]
    else:
        return all_dcs


def select_subset(data: Dict, mode: str) -> List[Dict]:
    """Seleciona um subset de data centers para testes."""
    all_dcs = data['data_centers']
    
    if mode == 'small':
        # 3 DCs: 1 Ascenty, 1 Scala, 1 outro
        return [all_dcs[0], all_dcs[8], all_dcs[15]]  # HC2, SGRUTB01, HostDime
    elif mode == 'medium':
        # 8 DCs: amostra geográfica
        indices = [0, 2, 4, 8, 10, 13, 14, 16]  # Mix de estados e operadores
        return [all_dcs[i] for i in indices]
    elif mode == 'full':
        return all_dcs
    else:
        return []


def generate_test_report(selected: List[Dict]):
    """Gera um relatório dos DCs selecionados para teste."""
    df = pd.DataFrame(selected)
    print("\n" + "="*100)
    print("DATA CENTERS SELECIONADOS PARA TESTE")
    print("="*100)
    print(df[['facility_id', 'name', 'operator', 'state', 'city', 'confidence']].to_string(index=False))
    print(f"\nTotal selecionado: {len(df)}")
    print(f"Por Operador:\n{df['operator'].value_counts().to_string()}")
    print(f"\nPor Estado:\n{df['state'].value_counts().to_string()}")
    print("="*100 + "\n")


def run_step3_for_site(name: str, construction_year: int, operation_year: int) -> bool:
    """Executa Step 3 para um data center específico."""
    csv_path = f'data/processed/{name}_cobertura_por_ano.csv'
    if not Path(csv_path).exists():
        print(f"  ️  Arquivo não encontrado: {csv_path}")
        return False
    
    try:
        cmd = [
            'python', 'scripts/step3_analise_impacto_temporal.py',
            '--csv', csv_path,
            '--name', name,
            '--construction-year', str(construction_year),
            '--operation-year', str(operation_year),
            '--output-dir', 'data/processed'
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   Step 3 concluído para {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   Erro ao executar Step 3: {e}")
        return False


def run_portfolio_analysis(selected: List[Dict]) -> bool:
    """Executa análise de portfólio para todos os DCs selecionados."""
    site_years = ';'.join([
        f"{dc['name']}={dc['construction_year']}:{dc['operation_year']}"
        for dc in selected
    ])
    
    try:
        cmd = [
            'python', 'scripts/step3_analise_impacto_temporal.py',
            '--portfolio',
            '--processed-dir', 'data/processed',
            '--site-years', site_years,
            '--output-dir', 'data/processed'
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(" Análise de portfólio concluída")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Erro ao executar análise de portfólio: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--list', action='store_true', help='Lista todos os data centers disponíveis')
    parser.add_argument('--test-small', action='store_true', help='Testa 3 data centers (quick test)')
    parser.add_argument('--test-medium', action='store_true', help='Testa 8 data centers (medium test)')
    parser.add_argument('--test-full', action='store_true', help='Testa todos os data centers (full test)')
    parser.add_argument('--test-operators', help='Filtra por operador: --test-operators Ascenty')
    parser.add_argument('--test-state', help='Filtra por estado: --test-state SP')
    parser.add_argument('--test-ids', help='Filtra por IDs: --test-ids 1,3,5,7')
    parser.add_argument('--skip-step3', action='store_true', help='Pula Step 3 (assume que Step 1 e 2 já foram executados)')
    parser.add_argument('--portfolio-only', action='store_true', help='Só executa análise de portfólio (Step 3)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        data = load_data_centers()
    except FileNotFoundError as e:
        print(f" Erro: {e}")
        return 1
    
    # Opção 1: Listar todos
    if args.list:
        print_data_centers(data)
        return 0
    
    # Opção 2: Selecionar subset
    selected = []
    if args.test_small:
        selected = select_subset(data, 'small')
        print("[QUICK] Modo: Quick Test (3 data centers)")
    elif args.test_medium:
        selected = select_subset(data, 'medium')
        print("[MEDIUM] Modo: Medium Test (8 data centers)")
    elif args.test_full:
        selected = select_subset(data, 'full')
        print("[FULL] Modo: Full Test (todos os data centers)")
    elif args.test_operators:
        selected = filter_data_centers(data, 'operator', args.test_operators)
        print(f"[FILTER] Modo: Filtrado por Operador ({args.test_operators})")
    elif args.test_state:
        selected = filter_data_centers(data, 'state', args.test_state)
        print(f"[FILTER] Modo: Filtrado por Estado ({args.test_state})")
    elif args.test_ids:
        selected = filter_data_centers(data, 'ids', args.test_ids)
        print(f"[FILTER] Modo: Filtrado por IDs ({args.test_ids})")
    else:
        print(" Nenhuma opção selecionada. Use --help para ver as opções.")
        return 1
    
    if not selected:
        print(" Nenhum data center selecionado após filtro.")
        return 1
    
    generate_test_report(selected)
    
    # Executar Step 3 para cada site
    if not args.portfolio_only:
        if not args.skip_step3:
            print(" Executando Step 3 para cada data center...\n")
            for dc in selected:
                print(f"Processando: {dc['name']}")
                run_step3_for_site(
                    dc['name'],
                    dc['construction_year'],
                    dc['operation_year']
                )
    
    # Executar análise de portfólio
    print("\n Executando análise de portfólio consolidada...")
    run_portfolio_analysis(selected)
    
    # Exibir resultado final
    ranking_file = Path('data/processed/portfolio_impact_ranking.csv')
    if ranking_file.exists():
        print("\n" + "="*100)
        print("RANKING CONSOLIDADO DE IMPACTO")
        print("="*100)
        ranking_df = pd.read_csv(ranking_file)
        print(ranking_df[['rank', 'facility_name', 'impact_score', 'delta_construcao', 'delta_vegetacao']].to_string(index=False))
        print("="*100 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
