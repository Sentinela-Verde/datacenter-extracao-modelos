"""Step 2 — rotulagem, treino (Random Forest + rede neural) e classificação da série temporal.

Reaproveita os GeoTIFFs e o `metadata.json` gerados por `src/extraction.py` (Step 1) e os
índices espectrais de `src/indices.py`.

Correções aplicadas em relação ao notebook original (ver "Bugs a corrigir" em
`modificacoes_projeto.md`):
- `matplotlib.colors`/`matplotlib.patches` importados aqui no topo do módulo (faltavam na célula
  de imports do notebook).
- `plot_mask_overlay` agora é definida antes de qualquer célula que a chame, pois todo o módulo é
  carregado de uma vez (no notebook original a célula que a usava vinha antes da que a definia).
"""
import json
import os

import ee
import geemap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import rasterio
from rasterio import features
from rasterio.warp import transform_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# Diretórios padronizados (ver src/extraction.py — antes havia o typo 'imanges_satelite' no step1
# vs. 'imagens_satelite' no step2). RAW_DIR guarda os GeoTIFFs Sentinel-2 + metadata.json;
# LABELS_DIR guarda os rótulos de referência (WorldCover + máscara de vias).
RAW_DIR = 'data/raw'
LABELS_DIR = 'data/labels'

CLASS_NAMES = ['Vegetação', 'Água', 'Construção', 'Estrada', 'Outro']
CLASS_COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#7f8c8d', '#f1c40f']

WORLDCOVER_MAP = {
    10: 0, 20: 0, 30: 0, 40: 0, 95: 0, 100: 0,  # vegetação
    80: 1, 90: 1,                                # água
    50: 2,                                        # construído
    60: 4, 70: 4,                                 # outro / solo exposto
}


def load_metadata(name_datacenter, out_dir=RAW_DIR):
    """Lê o metadata.json gerado pelo step1 (bandas, buffer, escala, CRS etc.)."""
    path = os.path.join(out_dir, f'{name_datacenter}_metadata.json')
    with open(path) as f:
        return json.load(f)


def tif_path(name_datacenter, year, out_dir=RAW_DIR):
    return os.path.join(out_dir, f'{name_datacenter}_{year}.tif')


def export_worldcover_labels(meta, out_dir=LABELS_DIR):
    """Exporta o ESA/WorldCover/v200 (rótulo de referência) para a mesma região do site."""
    os.makedirs(out_dir, exist_ok=True)

    center_point = ee.Geometry.Point([meta['lon'], meta['lat']])
    region = center_point.buffer(meta['buffer_m']).bounds()
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().clip(region)

    label_path = os.path.join(out_dir, f"{meta['name_datacenter']}_worldcover.tif")
    geemap.ee_export_image(
        worldcover, filename=label_path, scale=meta['scale'],
        region=region, crs=meta['crs'], file_per_band=False,
    )
    return label_path


def remap_worldcover(label_path):
    """Remapeia as classes originais do WorldCover para as 5 classes do projeto."""
    with rasterio.open(label_path) as src:
        raw = src.read(1)
    remapped = np.full(raw.shape, 4, dtype=np.uint8)
    for original_val, new_val in WORLDCOVER_MAP.items():
        remapped[raw == original_val] = new_val
    return remapped


def get_road_mask(reference_tif_path, buffer_m=4):
    """Rasteriza a malha viária do OpenStreetMap (via osmnx) na grade do GeoTIFF de referência.

    Também serve de base para o indicador de distância a vias (Distance), aplicando
    `scipy.ndimage.distance_transform_edt` sobre o resultado — ver Step 3 em
    `modificacoes_projeto.md`.
    """
    with rasterio.open(reference_tif_path) as src:
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
        bounds = src.bounds

    west, south, east, north = transform_bounds(crs, 'EPSG:4326', *bounds)
    roads = ox.features_from_bbox(bbox=(west, south, east, north), tags={'highway': True})
    roads = roads[roads.geometry.type.isin(['LineString', 'MultiLineString'])]

    if roads.empty:
        print('Nenhuma via encontrada pelo OSM nessa área.')
        return np.zeros(shape, dtype=bool)

    utm_crs = roads.estimate_utm_crs()
    roads_buffered = roads.to_crs(utm_crs).buffer(buffer_m).to_crs(crs)
    road_mask = features.rasterize(
        [(geom, 1) for geom in roads_buffered if geom is not None],
        out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
    )
    return road_mask.astype(bool)


def build_label_raster(remapped_worldcover, road_mask):
    """Combina o WorldCover remapeado com a máscara de vias (classe 'Estrada' tem prioridade)."""
    labels = remapped_worldcover.copy()
    labels[road_mask] = 3
    return labels


def extract_training_samples(feature_stack, labels, nodata_mask, n_samples_per_class=3000, seed=42):
    """Amostra pixels balanceados por classe a partir do raster de rótulos, pra treino."""
    rng = np.random.default_rng(seed)
    n_bands = feature_stack.shape[0]
    flat_features = feature_stack.reshape(n_bands, -1).T
    flat_labels = labels.ravel()
    flat_valid = ~nodata_mask.ravel()

    X_list, y_list = [], []
    for cls in np.unique(flat_labels):
        idx = np.where((flat_labels == cls) & flat_valid)[0]
        if len(idx) == 0:
            continue
        n = min(n_samples_per_class, len(idx))
        chosen = rng.choice(idx, size=n, replace=False)
        X_list.append(flat_features[chosen])
        y_list.append(flat_labels[chosen])

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f'Amostras de treino: {X.shape[0]} pixels')
    for cls in np.unique(y):
        print(f'  {CLASS_NAMES[cls]}: {(y == cls).sum()} amostras')
    return X, y


def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, n_jobs=-1,
        random_state=42, class_weight='balanced',
    )
    rf.fit(X_train, y_train)
    print(classification_report(y_test, rf.predict(X_test), target_names=CLASS_NAMES, zero_division=0))
    return rf


def train_neural_network(X_train, y_train, X_test, y_test, n_classes=5):
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_s, y_train, validation_split=0.15, epochs=30, batch_size=256, verbose=1)

    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return model, scaler


def classify_image(feature_stack, nodata_mask, model, scaler=None):
    n_bands, h, w = feature_stack.shape
    flat = feature_stack.reshape(n_bands, -1).T
    if scaler is not None:
        pred = np.argmax(model.predict(scaler.transform(flat), batch_size=4096, verbose=0), axis=1)
    else:
        pred = model.predict(flat)
    classified = pred.reshape(h, w).astype(np.uint8)
    classified[nodata_mask] = 255
    return classified


def compute_class_percentages(classified, pixel_size_m=10):
    valid = classified[classified != 255]
    total = valid.size
    pixel_area_km2 = (pixel_size_m ** 2) / 1e6
    return {
        name: {
            'percentual': round(100 * (valid == cls).sum() / total, 2) if total else 0,
            'area_km2': round((valid == cls).sum() * pixel_area_km2, 4),
        }
        for cls, name in enumerate(CLASS_NAMES)
    }


def plot_timeseries(df, name_datacenter, modelo='Random Forest', salvar_em=None, mostrar=True):
    """Plota a evolução do % de área por classe ao longo dos anos."""
    subset = df[df['modelo'] == modelo]
    fig, ax = plt.subplots(figsize=(10, 6))
    for classe, cor in zip(CLASS_NAMES, CLASS_COLORS):
        data = subset[subset['classe'] == classe]
        ax.plot(data['ano'], data['percentual'], marker='o', label=classe, color=cor)

    ax.set_title(f'Evolução da cobertura do solo — {name_datacenter} ({modelo})')
    ax.set_xlabel('Ano')
    ax.set_ylabel('% da área')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if salvar_em:
        plt.savefig(salvar_em, dpi=150, bbox_inches='tight')
        print(f'Salvo: {salvar_em}')

    if mostrar:
        plt.show()
    else:
        plt.close(fig)


def plot_mask_overlay(feature_stack, classified, titulo='', alpha=0.5, salvar_em=None, mostrar=True):
    """Sobrepõe a máscara de classificação (semi-transparente) à composição RGB do site."""
    red, green, blue = feature_stack[2], feature_stack[1], feature_stack[0]
    rgb = np.dstack([red, green, blue])
    rgb = np.clip(rgb / 0.3, 0, 1)

    cmap = mcolors.ListedColormap(CLASS_COLORS)
    bounds = list(range(len(CLASS_NAMES) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    mask = np.ma.masked_where(classified == 255, classified)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb)
    ax.imshow(mask, cmap=cmap, norm=norm, alpha=alpha)
    ax.set_title(titulo)
    ax.axis('off')

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1))

    if salvar_em:
        plt.savefig(salvar_em, dpi=150, bbox_inches='tight')
        print(f'Salvo: {salvar_em}')

    if mostrar:
        plt.show()
    else:
        plt.close(fig)
