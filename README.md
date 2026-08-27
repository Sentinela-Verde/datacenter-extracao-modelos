# Segmentação Semântica Supervisionada de Data Centers via Imagens de Satélite

Projeto do MBA em Engenharia de Dados (Mackenzie) que usa imagens de satélite Sentinel-2 para
monitorar, ao longo do tempo, a expansão da área construída de data centers no Brasil. O pipeline
extrai séries temporais de imagens via Google Earth Engine, gera rótulos de referência (uso e
cobertura do solo + malha viária) e treina classificadores supervisionados (Random Forest e uma
rede neural densa) para segmentar cada pixel em uma de cinco classes de cobertura do solo.

## Estrutura do projeto

```
.
├── principais_datacentes.txt                       # Lista dos data centers e suas coordenadas
├── step1_extracao_imagens_satelite_datacente.ipynb  # Extração das imagens Sentinel-2 (Earth Engine)
├── step2_classificacao_imagens.ipynb                # Rotulagem, treino e classificação (RF + rede neural)
├── imanges_satelite/                                # Saída do step1: GeoTIFFs, metadata e labels
├── imagens_jpg/                                      # Saída do step2: composições RGB e overlays de máscara
├── requirements.txt
├── .env.example                                      # Modelo de variáveis de ambiente (copie para .env)
└── .gitignore
```

> As pastas `imanges_satelite/` e `imagens_jpg/` são geradas pelos notebooks e versionadas no
> repositório; rodar o pipeline abaixo as regenera/atualiza.

## Pipeline

### 1. `step1_extracao_imagens_satelite_datacente.ipynb`

- Autentica no Google Earth Engine (`ee.Initialize`).
- Usa a coleção `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance) e aplica uma máscara
  de nuvens/cirros a partir da banda `QA60`.
- Para cada data center (nome, latitude, longitude) e cada ano de uma lista (`year_list`), monta um
  compósito da mediana da coleção filtrada por data e cobertura de nuvens, recorta um buffer ao
  redor do ponto e exporta um GeoTIFF com as bandas `B2, B3, B4, B8, B11, B12` (10 m de resolução)
  para `imanges_satelite/`.
- Também converte os GeoTIFFs em composições RGB (`B4, B3, B2`) salvas como JPG em `imagens_jpg/`
  para inspeção visual rápida.

### 2. `step2_classificacao_imagens.ipynb`

- Carrega o `metadata.json` gerado no step1 (bandas, buffer, escala, CRS etc.).
- **Rótulos de referência:**
  - Exporta o `ESA/WorldCover/v200` (mapa global de cobertura do solo) para a mesma região e o
    remapeia para 5 classes: `Vegetação`, `Água`, `Construção`, `Estrada`, `Outro`.
  - Complementa com a malha viária do OpenStreetMap (via `osmnx`), rasterizando as vias com um
    buffer para gerar a classe `Estrada`.
- **Features:** empilha as bandas do Sentinel-2 e calcula índices espectrais — NDVI (vegetação),
  NDWI (água) e NDBI (área construída) — como features adicionais por pixel.
- **Amostragem:** extrai amostras balanceadas por classe a partir do raster de rótulos para montar
  o conjunto de treino/teste.
- **Modelos treinados:**
  - `RandomForestClassifier` (scikit-learn, 300 árvores, `class_weight='balanced'`).
  - Rede neural densa (`tensorflow.keras`: Dense → Dropout → Dense → Dropout → Softmax) com
    features padronizadas via `StandardScaler`.
- **Classificação da série temporal:** aplica os modelos treinados a todos os anos disponíveis,
  calcula o percentual de área por classe e a área em km², e plota a evolução da cobertura do solo
  ao longo dos anos.
- **Visualização:** gera overlays (imagem RGB + máscara de classificação semi-transparente) para
  cada ano, salvos em `imagens_jpg/overlayer/`.

## Dados

`principais_datacentes.txt` lista os 10 principais data centers considerados (nome, endereço e
coordenadas), incluindo o data center Ascenty em Vinhedo/SP — atualmente o único processado pelos
notebooks (`name_datacenter = 'Ascenty_Vinhedo'`), com série temporal de imagens de 2016 a 2026.

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
da variável de ambiente `EE_PROJECT` (os notebooks carregam automaticamente um arquivo `.env` na
raiz do projeto via `python-dotenv`, se ele existir).

1. Copie o modelo e preencha com o ID do seu projeto:
   ```bash
   cp .env.example .env
   # edite .env e defina EE_PROJECT=seu-projeto-id
   ```
   (`.env` está no `.gitignore` e nunca deve ser commitado — só `.env.example`, que não tem
   segredos.)
2. Na primeira execução, descomente a linha `ee.Authenticate()` no início de cada notebook, rode-a
   e siga o fluxo de login no navegador.
3. Rode a célula de inicialização do Earth Engine — se `EE_PROJECT` não estiver definida, o
   notebook lança um erro explicativo em vez de seguir em frente.

## Como executar

1. Ajuste (se necessário) a lista de data centers/coordenadas e o intervalo de anos.
2. Execute `step1_extracao_imagens_satelite_datacente.ipynb` para baixar as imagens Sentinel-2.
3. Execute `step2_classificacao_imagens.ipynb` para gerar rótulos, treinar os modelos e produzir os
   mapas de classificação e os gráficos de evolução temporal.
