# Documentação da Extração de Dados — Data Centers no Brasil

## Visão geral

Este documento descreve o processo de coleta de dados sobre data centers e facilities de interconexão no Brasil, incluindo a fonte utilizada, a justificativa da escolha, a metodologia de extração via API, os campos coletados e as limitações conhecidas do conjunto de dados resultante.

O objetivo inicial do projeto era obter uma lista de data centers brasileiros com nome, operadora, localização e coordenadas geográficas. A primeira tentativa foi um scraping direto do site DataCenterMap (datacentermap.com/brazil/), mas essa abordagem esbarrou em bloqueios de taxa de requisição (erro HTTP 429) impostos pelo site a acessos automatizados. Diante disso, o projeto migrou para o PeeringDB, uma fonte de dados aberta e com API pública documentada, descrita em detalhe a seguir.

## Fonte dos dados: PeeringDB

O PeeringDB (peeringdb.com) é o banco de dados de origem utilizado nesta extração. Segundo a descrição oficial publicada pela própria organização em seu site, o PeeringDB é definido como:

> "Um banco de dados de redes disponível gratuitamente e mantido pelo usuário e o local ideal para dados de interconexão. O banco de dados facilita a interconexão global de redes em Internet Exchanges (IXPs), centros de dados (DCs) e outros PoPs de interconexão e é a primeira parada na tomada de decisões de interconexão."

O projeto se descreve como uma iniciativa sem fins lucrativos, dirigida pela comunidade, administrada e promovida por voluntários, funcionando como ferramenta pública para o crescimento da Internet. O site também explica que, embora tenha nascido como uma ferramenta voltada a peering entre redes, o escopo do banco de dados se expandiu ao longo dos anos para cobrir "todos os tipos de dados de interligação para redes, nuvens, serviços e empresas, bem como instalações de interligação que estão em desenvolvimento na fronteira da Internet."

O PeeringDB é mantido pelos próprios participantes do mercado: segundo a organização, quase um terço dos Números de Sistema Autônomo (ASNs) registrados no mundo cadastram seus próprios dados de interconexão na plataforma, e centros de dados e Internet Exchanges também cadastram e mantêm suas próprias informações para ganhar visibilidade junto a redes que buscam se conectar. Essa é uma diferença importante em relação a um site como o DataCenterMap: os registros do PeeringDB tendem a ser mantidos pelos próprios operadores das facilities, e não por um agregador comercial terceiro, o que reduz o risco de dados desatualizados e evita as barreiras técnicas anti-scraping que motivaram a mudança de fonte neste projeto. Em contrapartida, isso também significa que o PeeringDB só contém facilities cujos operadores optaram por se cadastrar — não é um censo completo de toda a infraestrutura de data centers do país.

O acesso aos dados foi feito por meio da API pública e documentada do PeeringDB (peeringdb.com/apidocs/), que devolve os registros em formato JSON e não exige autenticação para uso de leitura, respeitados os limites de requisição descritos adiante. Isso caracteriza um uso de API oficial e documentada, e não um scraping de HTML.

## Metodologia de extração

A extração foi feita em Python, com as bibliotecas `requests` (para chamadas HTTP) e `pandas` (para estruturação tabular dos dados).

O endpoint principal utilizado foi:

```
GET https://www.peeringdb.com/api/fac?country=BR&depth=2
```

O parâmetro `country=BR` filtra o resultado para facilities localizadas no Brasil. O parâmetro `depth=2` instrui a API a incluir, junto de cada registro, campos convenientes de entidades relacionadas — como o nome da organização proprietária (`org_name`) — evitando a necessidade de requisições adicionais para resolver esses dados.

A resposta em JSON foi achatada com `pandas.json_normalize()`, transformando qualquer campo aninhado (por exemplo, o objeto `org`) em colunas próprias (`org_name`, `org_website`, etc.). Campos que vêm como listas (por exemplo, arrays de IDs relacionados) foram convertidos para texto simples, unindo os valores com ponto e vírgula, de modo a caber corretamente em uma célula de planilha ou CSV.

Foi avaliada também uma etapa opcional de enriquecimento, que buscaria, para cada facility, a lista nominal de redes/operadoras presentes nela através do endpoint `/api/netfac`. Essa etapa foi descontinuada nesta versão da extração: como ela exige uma requisição HTTP por facility, o volume de chamadas (uma para cada uma das cerca de 250 facilities brasileiras) ultrapassou o limite de taxa da API e a maior parte das chamadas retornou erro 429. Optou-se por não seguir com essa abordagem, já que os campos-resumo `net_count`, `ix_count` e `carrier_count`, já presentes na tabela principal, cobrem a mesma necessidade analítica de forma agregada, sem risco de bloqueio.

### Limite de requisições (rate limit)

De acordo com a documentação oficial do PeeringDB, usuários anônimos (sem chave de API) estão sujeitos a um limite de 20 requisições por minuto por endereço IP, com restrições ainda mais rígidas para requisições idênticas repetidas. Usuários autenticados com chave de API têm um limite maior, de 40 requisições por minuto. A extração principal deste projeto respeita esse limite por se tratar de uma única chamada; qualquer extensão futura que exija múltiplas chamadas (como o enriquecimento por facility mencionado acima) deve agrupar identificadores usando o filtro `campo__in=valor1,valor2,...` — suportado pela API para consultas em lote de até aproximadamente 150 valores por chamada — em vez de uma chamada por registro.

## Campos coletados

A tabela abaixo resume os principais campos presentes na extração, conforme retornados pela API para cada facility:

| Campo | Descrição |
|---|---|
| `id` | Identificador único da facility no PeeringDB |
| `name` | Nome da facility/data center |
| `aka` | Nome alternativo ou apelido, quando existir |
| `org_id`, `org_name` | Identificador e nome da organização proprietária/operadora |
| `address1`, `address2` | Endereço completo |
| `city`, `state`, `zipcode`, `country` | Localização administrativa |
| `latitude`, `longitude` | Coordenadas geográficas da facility |
| `website` | Site institucional |
| `tech_email`, `tech_phone` | Contato técnico |
| `sales_email`, `sales_phone` | Contato comercial |
| `net_count` | Número de redes presentes na facility |
| `ix_count` | Número de Internet Exchanges presentes na facility |
| `carrier_count` | Número de operadoras de transporte/fibra presentes |
| `property` | Regime de propriedade do imóvel (ex.: "Owner") |
| `status` | Situação do registro (ex.: "ok") |
| `notes` | Observações livres cadastradas pelo operador |
| `created`, `updated` | Datas de criação e última atualização do registro |

O conjunto de dados final não se limita a essa lista: como a extração usa `pandas.json_normalize()` sobre a resposta completa da API, qualquer campo adicional retornado pela API é automaticamente incorporado à tabela, sem necessidade de ajuste manual do código.

## Limitações conhecidas

O PeeringDB é uma base mantida voluntariamente pelos próprios operadores de rede e de facilities, o que traz vantagens de atualização e confiabilidade, mas também significa que a cobertura não é exaustiva: instalações cujos operadores nunca se cadastraram na plataforma — em especial data centers privados/corporativos de menor porte, sem foco em peering — não aparecem nos resultados. Por essa razão, o número de facilities obtido pode ser menor do que o listado por agregadores comerciais como o DataCenterMap, que tentam catalogar toda a infraestrutura do mercado independentemente de cadastro voluntário.

Adicionalmente, a política de uso aceitável do PeeringDB, conforme descrita pela própria organização, veda o uso comercial dos dados e desestimula contato não solicitado com os operadores listados a partir das informações do banco — um ponto relevante para qualquer uso posterior desta extração além de fins analíticos internos.

## Como reproduzir a extração

1. Instalar as dependências: `pip install requests pandas` (ou `%pip install requests pandas` em um notebook Jupyter).
2. Executar a chamada HTTP ao endpoint `GET https://www.peeringdb.com/api/fac?country=BR&depth=2`.
3. Extrair a lista de registros do campo `data` da resposta JSON.
4. Achatar os registros em uma tabela com `pandas.json_normalize()`, convertendo campos do tipo lista/dicionário em texto.
5. Exportar a tabela final para CSV.

## Referências

- [PeeringDB — página institucional](https://www.peeringdb.com/)
- [PeeringDB — documentação da API](https://www.peeringdb.com/apidocs/)
- [PeeringDB — Working within PeeringDB's query limits](https://docs.peeringdb.com/howto/work_within_peeringdbs_query_limits/)
- [DataCenterMap — Brasil (fonte descartada por bloqueio de scraping)](https://www.datacentermap.com/brazil/)
