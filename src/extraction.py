"""Step 1 — extração da série temporal Sentinel-2 via Google Earth Engine.

Funções reaproveitadas por `notebooks/step1_extracao_imagens_satelite.ipynb` e por
`scripts/step1_extracao_imagens_satelite.py`, e reutilizáveis para extrair os sítios de controle
e os demais data centers listados em `principais_datacentes.txt` — basta chamar
`extract_datacenter_timeseries` de novo com outro `name_datacenter`/`lat`/`lon`.
"""
import json
import os

import ee
import geemap
import numpy as np
import rasterio

DEFAULT_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

# Nome padronizado da pasta de saída (antes havia inconsistência entre 'imanges_satelite' no
# step1 e 'imagens_satelite' no step2 — ver seção "Bugs a corrigir" em modificacoes_projeto.md).
RAW_DIR = 'data/raw'


def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask).divide(10000)


def extract_datacenter_timeseries(
    name_datacenter,
    lat,
    lon,
    year_list,
    month_start='05-01',
    month_end='07-30',
    cloud_pct=5,
    buffer_m=3000,
    scale=10,
    bands=None,
    out_dir=RAW_DIR,
):
    """Extrai, para cada ano de `year_list`, um compósito Sentinel-2 (mediana) em torno de
    (lat, lon), exporta um GeoTIFF por ano em `out_dir` e salva um `metadata.json` com os
    parâmetros usados (reaproveitado pelo step2/step3 para saber bandas, buffer, escala, CRS).
    """
    bands = bands or DEFAULT_BANDS
    os.makedirs(out_dir, exist_ok=True)

    for year in year_list:
        dataset = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(f'{year}-{month_start}', f'{year}-{month_end}')
            # Pre-filter to get less cloudy granules.
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
            .map(mask_s2_clouds)
        )

        # Cria um quadrado de ~5km ao redor do ponto central
        center_point = ee.Geometry.Point([lon, lat])
        region = center_point.buffer(buffer_m).bounds()

        # --- Preparar a imagem final para exportação ---
        image_to_export = dataset.mean().select(bands)

        # --- Baixar direto para a máquina local ---
        geemap.ee_export_image(
            image_to_export,
            filename=os.path.join(out_dir, f'{name_datacenter}_{year}.tif'),
            scale=scale,
            region=region,
            file_per_band=False,
        )

        print(f'[{name_datacenter} {year}] Download concluído.')

    # --- Salva os metadados usados, pra reaproveitar depois na classificação ---
    metadata = {
        'name_datacenter': name_datacenter,
        'lat': lat,
        'lon': lon,
        'year_list': year_list,
        'bands': bands,
        'buffer_m': buffer_m,
        'scale': scale,
        'crs': 'EPSG:4326',
    }
    metadata_path = os.path.join(out_dir, f'{name_datacenter}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nMetadados salvos em {metadata_path}')
    return metadata_path


def tif_to_rgb(tif_path, red_index=3, green_index=2, blue_index=1, vis_max=0.3):
    """Lê um GeoTIFF Sentinel-2 (bandas na ordem B2,B3,B4,B8,B11,B12) e devolve um array RGB
    (H, W, 3) já normalizado (clip 0-1) para plot com matplotlib."""
    with rasterio.open(tif_path) as src:
        # Bandas na ordem exportada: B2,B3,B4,B8,B11,B12.
        # Para RGB "natural", precisamos de B4(vermelho), B3(verde), B2(azul).
        red = src.read(red_index)
        green = src.read(green_index)
        blue = src.read(blue_index)

    rgb = np.dstack([red, green, blue])
    return np.clip(rgb / vis_max, 0, 1)  # normaliza pro range visível (mesmo min/max da visualization do GEE)


def export_rgb_jpgs(pasta_entrada=RAW_DIR, pasta_saida='imagens_jpg'):
    """Converte todos os GeoTIFFs de `pasta_entrada` em composições RGB salvas como JPG em
    `pasta_saida`, para inspeção visual rápida da série temporal."""
    import matplotlib.pyplot as plt

    os.makedirs(pasta_saida, exist_ok=True)

    for nome_arquivo in sorted(os.listdir(pasta_entrada)):
        if not nome_arquivo.endswith('.tif'):
            continue

        path = os.path.join(pasta_entrada, nome_arquivo)
        rgb = tif_to_rgb(path)

        nome_saida = os.path.splitext(nome_arquivo)[0] + '.jpg'
        path_saida = os.path.join(pasta_saida, nome_saida)

        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.title(nome_arquivo)
        plt.axis('off')
        plt.savefig(path_saida, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'Salvo: {path_saida}')
