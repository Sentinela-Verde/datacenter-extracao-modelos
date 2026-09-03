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
import sys

import ee
import geemap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
import tensorflow as tf
from rasterio import features
from rasterio.warp import transform_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    cohen_kappa_score, jaccard_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
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
    """Treinar Random Forest com hiperparâmetros otimizados."""
    print("\n" + "="*80)
    print("RANDOM FOREST - TREINO E AVALIAÇÃO")
    print("="*80 + "\n")
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Métricas detalhadas
    from sklearn.metrics import f1_score, cohen_kappa_score
    accuracy = (y_pred == y_test).mean()
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score (ponderado): {f1_weighted:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    return rf


def train_xgboost(X_train, y_train, X_test, y_test):
    """Treinar XGBoost (Gradient Boosting melhorado)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("⚠️ XGBoost não instalado. Instalando...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "xgboost"], check=False)
        from xgboost import XGBClassifier
    
    print("\n" + "="*80)
    print("XGBOOST - TREINO E AVALIAÇÃO")
    print("="*80 + "\n")
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    y_pred = xgb.predict(X_test)
    
    # Métricas detalhadas
    from sklearn.metrics import f1_score, cohen_kappa_score
    accuracy = (y_pred == y_test).mean()
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score (ponderado): {f1_weighted:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    return xgb


def train_neural_network(X_train, y_train, X_test, y_test, n_classes=5):
    """Treinar rede neural com melhores práticas."""
    print("\n" + "="*80)
    print("REDE NEURAL DENSA - TREINO E AVALIAÇÃO")
    print("="*80 + "\n")
    
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping e learning rate reduction
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )
    
    model.fit(
        X_train_s, y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=256,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
    
    # Métricas detalhadas
    from sklearn.metrics import f1_score, cohen_kappa_score
    accuracy = (y_pred == y_test).mean()
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"\nAcurácia: {accuracy:.4f}")
    print(f"F1-Score (ponderado): {f1_weighted:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print("\nRelatório de Classificação:")
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


# ============================================================
# VALIDAÇÃO ROBUSTA E BALANCEAMENTO
# ============================================================

def apply_smote(X_train, y_train, random_state=42):
    """Aplicar SMOTE para balancear classes (requer imbalanced-learn)."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("⚠️ imbalanced-learn não instalado. Instalando...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "imbalanced-learn"], check=False)
        from imblearn.over_sampling import SMOTE
    
    print("\n" + "="*80)
    print("SMOTE - BALANCEAMENTO DE CLASSES")
    print("="*80 + "\n")
    
    # Antes
    unique, counts = np.unique(y_train, return_counts=True)
    print("Distribuição ANTES do SMOTE:")
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(y_train)
        print(f"  {CLASS_NAMES[cls]:<20}: {count:>6} amostras ({pct:>5.1f}%)")
    
    # Aplicar SMOTE
    smote = SMOTE(k_neighbors=5, random_state=random_state, n_jobs=-1)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    # Depois
    unique, counts = np.unique(y_balanced, return_counts=True)
    print("\nDistribuição DEPOIS do SMOTE:")
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(y_balanced)
        print(f"  {CLASS_NAMES[cls]:<20}: {count:>6} amostras ({pct:>5.1f}%)")
    
    print(f"\nAmostras adicionadas: {len(X_balanced) - len(X_train)}")
    print()
    
    return X_balanced, y_balanced


def train_with_kfold(model, X, y, n_splits=5, random_state=42):
    """Treinar com K-Fold Cross-Validation para avaliação robusta."""
    print("\n" + "="*80)
    print(f"K-FOLD CROSS-VALIDATION ({n_splits} folds)")
    print("="*80 + "\n")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    trained_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_idx + 1}/{n_splits}...", end=" ")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Treinar
        if hasattr(model, 'fit'):  # Verificar se é um modelo sklearn
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
        else:
            print("❌ Modelo não suportado")
            continue
        
        # Calcular métricas
        accuracy = (y_pred == y_val_fold).mean()
        f1_w = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_val_fold, y_pred)
        jaccard = jaccard_score(y_val_fold, y_pred, average='weighted', zero_division=0)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': accuracy,
            'f1_weighted': f1_w,
            'kappa': kappa,
            'jaccard': jaccard
        })
        trained_models.append(model)
        
        print(f"✓ Acc={accuracy:.4f}, F1={f1_w:.4f}")
    
    # Resumo
    print("\n" + "="*80)
    print("RESUMO DOS FOLDS")
    print("="*80 + "\n")
    
    results_df = pd.DataFrame(fold_results)
    print(f"Acurácia:     {results_df['accuracy'].mean():.4f} (±{results_df['accuracy'].std():.4f})")
    print(f"F1 Ponderado: {results_df['f1_weighted'].mean():.4f} (±{results_df['f1_weighted'].std():.4f})")
    print(f"Kappa:        {results_df['kappa'].mean():.4f} (±{results_df['kappa'].std():.4f})")
    print(f"Jaccard (IoU):{results_df['jaccard'].mean():.4f} (±{results_df['jaccard'].std():.4f})")
    print()
    
    return results_df, trained_models


def plot_confusion_matrix(y_true, y_pred):
    """Plotar matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES,
           xlabel='Predito',
           ylabel='Verdadeiro')
    
    # Adicionar texto
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()
    
    return cm


def detailed_metrics_report(y_true, y_pred):
    """Gerar relatório detalhado de métricas."""
    print("\n" + "="*80)
    print("RELATÓRIO DETALHADO DE CLASSIFICAÇÃO")
    print("="*80 + "\n")
    
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    print("="*80)
    print("MÉTRICAS ADICIONAIS")
    print("="*80 + "\n")
    
    # Jaccard (IoU) por classe
    jaccard_per_class = jaccard_score(y_true, y_pred, average=None, zero_division=0)
    print("Jaccard Index (IoU) por classe:")
    for cls, score in enumerate(jaccard_per_class):
        print(f"  {CLASS_NAMES[cls]:<20}: {score:.4f}")
    
    print(f"\n  Média ponderada (mIoU): {jaccard_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\nCohen's Kappa: {kappa:.4f}")
    
    # Matriz de confusão
    plot_confusion_matrix(y_true, y_pred)
