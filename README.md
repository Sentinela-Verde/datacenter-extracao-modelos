# Segmentação Semântica Supervisionada de Data Centers via Imagens de Satélite

Projeto do MBA em Engenharia de Dados (Mackenzie) que usa imagens de satélite Sentinel-2 para
monitorar, ao longo do tempo, a expansão da área construída de data centers no Brasil. O pipeline
extrai séries temporais de imagens via Google Earth Engine, gera rótulos de referência (uso e
cobertura do solo + malha viária) e treina classificadores supervisionados (Random Forest e uma
rede neural densa) para segmentar cada pixel em uma de cinco classes de cobertura do solo. As
próximas etapas (ver `modificacoes_projeto.md`) consolidam índices espectrais e métricas adicionais
numa tabela única e constroem um indicador de impacto retrospectivo (diferença-em-diferenças).

## Estrutura do projeto

```
.
├── data/
│   ├── raw/                                          # Step 1: GeoTIFFs Sentinel-2 + metadata.json por site
│   ├── labels/                                        # Step 2: rótulos WorldCover
│   └── processed/                                     # Step 2+: tabelas de cobertura por ano e (futuro) metricas_painel.csv
├── notebooks/
│   ├── step1_extracao_imagens_satelite.ipynb          # Extração das imagens Sentinel-2 (Earth Engine)
│   └── step2_classificacao_imagens.ipynb              # Rotulagem, treino e classificação (RF + rede neural)
├── src/                                                # Funções reutilizáveis (usadas pelos notebooks e pelos scripts)
│   ├── extraction.py                                   # mask_s2_clouds, extract_datacenter_timeseries, export_rgb_jpgs
│   ├── indices.py                                      # NDVI/NDWI/NDBI/EVI/SAVI/BSI/MNDWI/IBI/NDMI, load_features
│   ├── classification.py                               # rotulagem (WorldCover + OSM), treino/aplicação RF e rede neural
│   └── impact.py                                       # placeholder do indicador de impacto (Step 3/4, ver modificacoes_projeto.md)
├── scripts/                                            # Equivalente em linha de comando dos notebooks
│   ├── step1_extracao_imagens_satelite.py
│   └── step2_classificacao_imagens.py
├── imagens_jpg/                                        # Composições RGB e overlays de máscara (saída visual do step1/step2)
├── principais_datacentes.txt                           # Lista dos data centers e suas coordenadas
├── modificacoes_projeto.md                             # Especificação técnica das próximas etapas (Step 3)
├── requirements.txt
├── .env.example                                        # Modelo de variáveis de ambiente (copie para .env)
└── .gitignore
```

> As pastas `data/` e `imagens_jpg/` são geradas pelo pipeline e versionadas no repositório;
> rodar os notebooks/scripts abaixo as regenera/atualiza.

## Pipeline

Cada etapa existe em duas formas equivalentes: um notebook em `notebooks/` (para explorar
interativamente, célula a célula) e um script em `scripts/` (para rodar via terminal, ex. em lote
para vários data centers). Ambos chamam as mesmas funções, definidas uma única vez em `src/`.

### 1. Extração (`src/extraction.py`)

- Autentica no Google Earth Engine (`ee.Initialize`).
- Usa a coleção `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance) e aplica uma máscara
  de nuvens/cirros a partir da banda `QA60` (`mask_s2_clouds`).
- Para cada data center (nome, latitude, longitude) e cada ano de uma lista (`year_list`), monta um
  compósito da mediana da coleção filtrada por data e cobertura de nuvens, recorta um buffer ao
  redor do ponto e exporta um GeoTIFF com as bandas `B2, B3, B4, B8, B11, B12` (10 m de resolução)
  para `data/raw/` (`extract_datacenter_timeseries`).
- Também converte os GeoTIFFs em composições RGB (`B4, B3, B2`) salvas como JPG em `imagens_jpg/`
  para inspeção visual rápida (`export_rgb_jpgs`).

```bash
python scripts/step1_extracao_imagens_satelite.py --name Ascenty_Vinhedo --lat -23.071035 --lon -47.011837
```

### 2. Classificação (`src/classification.py` + `src/indices.py`)

- Carrega o `metadata.json` gerado no step1 (bandas, buffer, escala, CRS etc.).
- **Rótulos de referência:**
  - Exporta o `ESA/WorldCover/v200` (mapa global de cobertura do solo) para `data/labels/` e o
    remapeia para 5 classes: `Vegetação`, `Água`, `Construção`, `Estrada`, `Outro`.
  - Complementa com a malha viária do OpenStreetMap (via `osmnx`), rasterizando as vias com um
    buffer para gerar a classe `Estrada`.
- **Features:** empilha as bandas do Sentinel-2 e calcula os índices espectrais NDVI, NDWI, NDBI,
  EVI, SAVI, BSI, MNDWI, IBI e NDMI como features adicionais por pixel (`load_features`).
- **Amostragem:** extrai amostras balanceadas por classe a partir do raster de rótulos para montar
  o conjunto de treino/teste.
- **Modelos treinados:**
  - `RandomForestClassifier` (scikit-learn, 300 árvores, `class_weight='balanced'`).
  - Rede neural densa (`tensorflow.keras`: Dense → Dropout → Dense → Dropout → Softmax) com
    features padronizadas via `StandardScaler`.
- **Classificação da série temporal:** aplica os modelos treinados a todos os anos disponíveis,
  calcula o percentual de área por classe e a área em km² (salvo em `data/processed/`), e plota a
  evolução da cobertura do solo ao longo dos anos.
- **Visualização:** gera overlays (imagem RGB + máscara de classificação semi-transparente) para
  cada ano, salvos em `imagens_jpg/`.

```bash
python scripts/step2_classificacao_imagens.py --name Ascenty_Vinhedo --ano-referencia 2024
```

### 3–4. Próximos passos

Extração completa das métricas adicionais (LST, Slope, Distance, anéis de distância) e o
indicador de impacto via diferença-em-diferenças — ver `modificacoes_projeto.md` para a
especificação completa. O placeholder `src/impact.py` documenta o plano dessas etapas.

## Dados

`principais_datacentes.txt` lista os 10 principais data centers considerados (nome, endereço e
coordenadas), incluindo o data center Ascenty em Vinhedo/SP — atualmente o único processado
(`name_datacenter = 'Ascenty_Vinhedo'`), com série temporal de imagens de 2016 a 2026.

## Requisitos

- Python 3.12+
- Conta no Google Earth Engine com um projeto no Google Cloud habilitado (necessário para
  `ee.Initialize(project=...)`)

### Instalação

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Autenticação no Earth Engine

O ID do projeto do Google Cloud usado pelo `ee.Initialize` **não fica fixo no código** — ele é lido
da variável de ambiente `EE_PROJECT` (os notebooks e scripts carregam automaticamente um arquivo
`.env` na raiz do projeto via `python-dotenv`, se ele existir).

1. Copie o modelo e preencha com o ID do seu projeto:
   ```bash
   cp .env.example .env
   # edite .env e defina EE_PROJECT=seu-projeto-id
   ```
   (`.env` está no `.gitignore` e nunca deve ser commitado — só `.env.example`, que não tem
   segredos.)
2. Na primeira execução, descomente a linha `ee.Authenticate()` no início de cada notebook (ou
   passe `--authenticate` ao rodar `scripts/step1_extracao_imagens_satelite.py`) e siga o fluxo de
   login no navegador.
3. Rode a célula/etapa de inicialização do Earth Engine — se `EE_PROJECT` não estiver definida, o
   código lança um erro explicativo em vez de seguir em frente.

## Como executar

Via notebooks (interativo):

1. Ajuste (se necessário) a lista de data centers/coordenadas e o intervalo de anos.
2. Execute `notebooks/step1_extracao_imagens_satelite.ipynb` para baixar as imagens Sentinel-2.
3. Execute `notebooks/step2_classificacao_imagens.ipynb` para gerar rótulos, treinar os modelos e
   produzir os mapas de classificação e os gráficos de evolução temporal.

Via linha de comando (útil para repetir o pipeline em outros sites, ex. os demais 9 data centers de
`principais_datacentes.txt` ou sítios de controle):

```bash
python scripts/step1_extracao_imagens_satelite.py --name <nome> --lat <lat> --lon <lon>
python scripts/step2_classificacao_imagens.py --name <nome> --ano-referencia <ano>
```

Rode `python scripts/step1_extracao_imagens_satelite.py --help` (ou `step2_...py --help`) para ver
todas as opções disponíveis.
