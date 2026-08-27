"""Step 2 — rotulagem, treino (RF + rede neural) e classificação da série temporal (linha de comando).

Equivalente em .py de `notebooks/step2_classificacao_imagens.ipynb`: exporta os rótulos de
referência (WorldCover + vias), treina os dois classificadores no ano de referência, aplica sobre
toda a série temporal e salva a tabela de % de área por classe/ano. Toda a lógica reutilizável está
em `src/classification.py` e `src/indices.py`.

Uso:
    python scripts/step2_classificacao_imagens.py
    python scripts/step2_classificacao_imagens.py --name Ascenty_Vinhedo --ano-referencia 2024
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.classification import (
    CLASS_NAMES,
    LABELS_DIR,
    RAW_DIR,
    build_label_raster,
    classify_image,
    compute_class_percentages,
    export_worldcover_labels,
    extract_training_samples,
    get_road_mask,
    load_metadata,
    plot_mask_overlay,
    plot_timeseries,
    remap_worldcover,
    tif_path,
    train_neural_network,
    train_random_forest,
)
from src.indices import load_features

PROCESSED_DIR = 'data/processed'
JPG_DIR = 'imagens_jpg'


def init_earth_engine():
    load_dotenv()
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
    parser.add_argument('--name', default='Ascenty_Vinhedo', help='Nome do data center/site (deve casar com o usado no step1)')
    parser.add_argument('--ano-referencia', type=int, default=2024, help='Ano usado para treinar os modelos')
    parser.add_argument('--raw-dir', default=RAW_DIR, help='Pasta com os GeoTIFFs + metadata.json do step1')
    parser.add_argument('--labels-dir', default=LABELS_DIR, help='Pasta de saída dos rótulos (WorldCover)')
    parser.add_argument('--processed-dir', default=PROCESSED_DIR, help='Pasta de saída da tabela de cobertura por ano')
    parser.add_argument('--jpg-dir', default=JPG_DIR, help='Pasta de saída dos overlays de classificação')
    parser.add_argument('--road-buffer-m', type=float, default=4, help='Buffer (m) aplicado às vias do OSM')
    parser.add_argument('--skip-overlays', action='store_true', help='Não gerar overlays JPG por ano')
    parser.add_argument('--show-plots', action='store_true', help='Mostrar os gráficos interativamente (plt.show)')
    return parser.parse_args()


def main():
    args = parse_args()
    init_earth_engine()

    meta = load_metadata(args.name, args.raw_dir)
    band_names = meta['bands']
    ref_tif = tif_path(args.name, args.ano_referencia, args.raw_dir)
    print(meta)

    # --- Rótulos de referência: WorldCover remapeado + máscara de vias (OSM) ---
    label_path = export_worldcover_labels(meta, args.labels_dir)
    remapped = remap_worldcover(label_path)
    road_mask = get_road_mask(ref_tif, buffer_m=args.road_buffer_m)
    labels = build_label_raster(remapped, road_mask)

    # --- Features (bandas + índices espectrais) do ano de referência e amostragem de treino ---
    feature_stack, nodata_mask = load_features(ref_tif, band_names)
    X, y = extract_training_samples(feature_stack, labels, nodata_mask)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # --- Treino dos dois classificadores ---
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    nn_model, scaler = train_neural_network(X_train, y_train, X_test, y_test)

    # --- Classifica toda a série temporal e monta a tabela de % de área por classe/ano ---
    os.makedirs(args.processed_dir, exist_ok=True)
    rows = []
    for year in meta['year_list']:
        path = tif_path(args.name, year, args.raw_dir)
        if not os.path.exists(path):
            print(f'[{year}] arquivo não encontrado, pulei.')
            continue

        fstack, nmask = load_features(path, band_names)
        classified_rf = classify_image(fstack, nmask, rf_model)
        classified_nn = classify_image(fstack, nmask, nn_model, scaler)

        pct_rf = compute_class_percentages(classified_rf)
        pct_nn = compute_class_percentages(classified_nn)

        for classe in CLASS_NAMES:
            rows.append({'ano': year, 'classe': classe, 'modelo': 'Random Forest',
                         'percentual': pct_rf[classe]['percentual'], 'area_km2': pct_rf[classe]['area_km2']})
            rows.append({'ano': year, 'classe': classe, 'modelo': 'Rede Neural',
                         'percentual': pct_nn[classe]['percentual'], 'area_km2': pct_nn[classe]['area_km2']})
        print(f'[{year}] classificado.')

        if not args.skip_overlays:
            os.makedirs(args.jpg_dir, exist_ok=True)
            plot_mask_overlay(
                fstack, classified_rf,
                titulo=f'{args.name} {year} - Random Forest', alpha=0.5,
                salvar_em=os.path.join(args.jpg_dir, f'{args.name}_{year}_overlay.jpg'),
                mostrar=False,
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.processed_dir, f'{args.name}_cobertura_por_ano.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nTabela salva em {csv_path}')

    plot_timeseries(
        df, args.name,
        salvar_em=os.path.join(args.processed_dir, f'{args.name}_evolucao.png'),
        mostrar=args.show_plots,
    )

    if not args.skip_overlays:
        print('Todas as máscaras geradas.')


if __name__ == '__main__':
    main()
