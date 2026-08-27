"""Step 1 — extração da série temporal Sentinel-2 para um data center (linha de comando).

Equivalente em .py de `notebooks/step1_extracao_imagens_satelite.ipynb`: autentica no Google Earth
Engine, extrai um GeoTIFF por ano num buffer ao redor de (lat, lon) e gera as composições RGB em
JPG. Toda a lógica reutilizável está em `src/extraction.py`.

Uso:
    python scripts/step1_extracao_imagens_satelite.py
    python scripts/step1_extracao_imagens_satelite.py --name Site_Controle_1 --lat -23.10 --lon -47.20 --ano-inicio 2016 --ano-fim 2026
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
from dotenv import load_dotenv

from src.extraction import RAW_DIR, export_rgb_jpgs, extract_datacenter_timeseries

DEFAULT_NAME = 'Ascenty_Vinhedo'
DEFAULT_LAT = -23.071035
DEFAULT_LON = -47.011837
DEFAULT_ANO_INICIO = 2016
DEFAULT_ANO_FIM = 2026


def init_earth_engine():
    load_dotenv()  # carrega variáveis de ambiente do arquivo .env, se existir
    ee_project = os.environ.get('EE_PROJECT')
    if not ee_project:
        raise RuntimeError(
            "Defina a variável de ambiente EE_PROJECT com o ID do seu projeto no Google Cloud "
            "antes de rodar este script (crie um arquivo .env a partir de .env.example com "
            "EE_PROJECT=seu-projeto-id)."
        )
    ee.Initialize(project=ee_project)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default=DEFAULT_NAME, help='Nome do data center/site (usado nos arquivos de saída)')
    parser.add_argument('--lat', type=float, default=DEFAULT_LAT)
    parser.add_argument('--lon', type=float, default=DEFAULT_LON)
    parser.add_argument('--ano-inicio', type=int, default=DEFAULT_ANO_INICIO)
    parser.add_argument('--ano-fim', type=int, default=DEFAULT_ANO_FIM)
    parser.add_argument('--buffer-m', type=int, default=3000, help='Raio do buffer (m) ao redor do ponto central')
    parser.add_argument('--out-dir', default=RAW_DIR, help='Pasta de saída dos GeoTIFFs + metadata.json')
    parser.add_argument('--jpg-dir', default='imagens_jpg', help='Pasta de saída das composições RGB')
    parser.add_argument('--skip-jpg', action='store_true', help='Não gerar as composições RGB em JPG')
    parser.add_argument('--authenticate', action='store_true', help='Roda ee.Authenticate() antes de inicializar (primeira execução)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.authenticate:
        ee.Authenticate()
    init_earth_engine()

    year_list = list(range(args.ano_inicio, args.ano_fim + 1))
    extract_datacenter_timeseries(
        args.name, args.lat, args.lon, year_list,
        buffer_m=args.buffer_m, out_dir=args.out_dir,
    )

    if not args.skip_jpg:
        export_rgb_jpgs(pasta_entrada=args.out_dir, pasta_saida=args.jpg_dir)


if __name__ == '__main__':
    main()
