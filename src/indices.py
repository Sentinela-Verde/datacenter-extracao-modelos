"""Índices espectrais calculados a partir das bandas Sentinel-2 (B2,B3,B4,B8,B11,B12).

Usado pelo Step 2 (classificação, via `load_features`) e reaproveitável pelo futuro Step 3 (tabela
de métricas/indicadores de impacto, ver `modificacoes_projeto.md`). Todos os índices abaixo são
calculáveis sem nova chamada ao Earth Engine — só reprocessam as bandas já extraídas pelo step1.
"""
import numpy as np
import rasterio

EPS = 1e-6


def compute_ndvi(b):
    """(B8-B4)/(B8+B4) — vegetação."""
    return (b['B8'] - b['B4']) / (b['B8'] + b['B4'] + EPS)


def compute_ndwi(b):
    """(B3-B8)/(B3+B8) — água."""
    return (b['B3'] - b['B8']) / (b['B3'] + b['B8'] + EPS)


def compute_ndbi(b):
    """(B11-B8)/(B11+B8) — área construída."""
    return (b['B11'] - b['B8']) / (b['B11'] + b['B8'] + EPS)


def compute_evi(b):
    """2.5*(B8-B4)/(B8+6*B4-7.5*B2+1) — vegetação, corrige efeito de solo/atmosfera."""
    return 2.5 * (b['B8'] - b['B4']) / (b['B8'] + 6 * b['B4'] - 7.5 * b['B2'] + 1 + EPS)


def compute_savi(b):
    """((B8-B4)/(B8+B4+0.5))*1.5 — vegetação ajustada pro solo exposto."""
    return ((b['B8'] - b['B4']) / (b['B8'] + b['B4'] + 0.5 + EPS)) * 1.5


def compute_bsi(b):
    """((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2)) — solo exposto."""
    return ((b['B11'] + b['B4']) - (b['B8'] + b['B2'])) / ((b['B11'] + b['B4']) + (b['B8'] + b['B2']) + EPS)


def compute_mndwi(b):
    """(B3-B11)/(B3+B11) — água, mais robusto que o NDWI em áreas urbanas."""
    return (b['B3'] - b['B11']) / (b['B3'] + b['B11'] + EPS)


def compute_ndmi(b):
    """(B8-B11)/(B8+B11) — umidade da vegetação (opcional)."""
    return (b['B8'] - b['B11']) / (b['B8'] + b['B11'] + EPS)


def compute_ibi(ndbi, savi, mndwi):
    """(NDBI-(SAVI+MNDWI)/2)/(NDBI+(SAVI+MNDWI)/2) — índice composto de área construída."""
    outros = (savi + mndwi) / 2
    return (ndbi - outros) / (ndbi + outros + EPS)


# Ordem fixa em que os índices são empilhados por `load_features` — mantenha estável, pois é usada
# para nomear as colunas de features em qualquer análise posterior (ex.: importância de variáveis).
SPECTRAL_INDEX_NAMES = ['NDVI', 'NDWI', 'NDBI', 'EVI', 'SAVI', 'BSI', 'MNDWI', 'IBI', 'NDMI']


def compute_spectral_indices(band_stack):
    """band_stack: dict {nome_da_banda: array 2D}. Retorna dict {nome_do_indice: array 2D}
    com todos os índices em `SPECTRAL_INDEX_NAMES`."""
    b = band_stack
    ndvi = compute_ndvi(b)
    ndwi = compute_ndwi(b)
    ndbi = compute_ndbi(b)
    evi = compute_evi(b)
    savi = compute_savi(b)
    bsi = compute_bsi(b)
    mndwi = compute_mndwi(b)
    ibi = compute_ibi(ndbi, savi, mndwi)
    ndmi = compute_ndmi(b)

    return {
        'NDVI': ndvi,
        'NDWI': ndwi,
        'NDBI': ndbi,
        'EVI': evi,
        'SAVI': savi,
        'BSI': bsi,
        'MNDWI': mndwi,
        'IBI': ibi,
        'NDMI': ndmi,
    }


def load_features(s2_tif_path, band_names):
    """Carrega um GeoTIFF Sentinel-2 e empilha as bandas brutas + todos os índices espectrais
    (NDVI, NDWI, NDBI, EVI, SAVI, BSI, MNDWI, IBI, NDMI) como features por pixel.

    Retorna (full_stack, nodata_mask), onde full_stack tem shape
    (n_bands + len(SPECTRAL_INDEX_NAMES), altura, largura). Os nomes das features, na mesma
    ordem, são `list(band_names) + SPECTRAL_INDEX_NAMES`.
    """
    with rasterio.open(s2_tif_path) as src:
        stack = src.read().astype(np.float32)
        nodata_mask = np.all(stack == 0, axis=0)

    b = dict(zip(band_names, stack))
    indices = compute_spectral_indices(b)
    index_stack = np.stack([indices[name] for name in SPECTRAL_INDEX_NAMES], axis=0)

    full_stack = np.concatenate([stack, index_stack], axis=0)
    return full_stack, nodata_mask
