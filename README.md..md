# TCGA-PAAD | Pancreatic Cancer AI Research

**Estudo piloto de análise clínica, sobrevida e inteligência artificial explicável em câncer de pâncreas**

![Status](https://img.shields.io/badge/status-estudo%20explorat%C3%B3rio-2563eb) ![Dataset](https://img.shields.io/badge/dados-TCGA--PAAD-0f766e) ![Python](https://img.shields.io/badge/linguagem-Python-3776ab) ![Área](https://img.shields.io/badge/%C3%A1rea-oncologia%20computacional-64748b)

> **Escopo científico:** repositório de um estudo piloto computacional com dados secundários públicos do projeto TCGA-PAAD, obtidos pelo Genomic Data Commons (GDC). Os resultados são preliminares e não constituem ferramenta clínica, validação de IA ou evidência causal.

## Visão geral

O projeto investiga características clínicas e desfechos de pacientes com câncer de pâncreas, estabelecendo uma base metodológica para pesquisas futuras sobre integração de dados clínicos e multiômicos e modelos interpretáveis de inteligência artificial.

**Objetivos do piloto:**

- Auditar, consolidar e caracterizar registros clínicos do TCGA-PAAD.
- Descrever idade ao diagnóstico, estadiamento patológico, situação vital e completude dos dados.
- Estimar a sobrevida global por Kaplan–Meier, respeitando eventos e censuras.
- Explorar associações entre características clínicas e risco de óbito por regressão de Cox.
- Examinar pressupostos, qualidade dos dados e sensibilidade das estimativas antes de etapas preditivas.

## Fonte dos dados e população

| Item | Descrição |
|:--|:--|
| Programa | The Cancer Genome Atlas (TCGA) |
| Projeto | TCGA-PAAD — Pancreatic Adenocarcinoma |
| Portal | [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/projects/TCGA-PAAD) |
| Modalidades do piloto | Dados clínicos e exploração inicial de RNA-seq |
| Unidade de análise clínica | Paciente único após consolidação |
| Tipo de estudo | Observacional retrospectivo, secundário e exploratório |

### Caracterização preliminar da coorte clínica

| Indicador | Resultado preliminar |
|:--|--:|
| Pacientes únicos | **185** |
| Idade média ao diagnóstico | **65,42 anos** |
| Idade mediana ao diagnóstico | **65,83 anos** |
| Falecidos na situação vital registrada | **100** |
| Vivos na situação vital registrada | **85** |
| Estágio patológico ausente | **3 (1,6%)** |

*Os números acima se referem à consolidação exploratória informada na execução local; não devem ser interpretados como estatísticas definitivas do projeto TCGA-PAAD. É necessário confirmar a origem dos tempos, os critérios de elegibilidade e a consistência entre arquivos antes de qualquer publicação científica.*

## Fluxo de análise

| Etapa | Procedimento | Situação |
|:--|:--|:--|
| 01 | Aquisição e organização dos arquivos clínicos e RNA-seq | Parcial; downloads com falhas de conexão |
| 02 | Inspeção das tabelas clínicas e identificação de variáveis | Executada |
| 03 | Consolidação de registros por paciente | Executada; requer auditoria |
| 04 | Análise exploratória e visualizações | Executada |
| 05 | Auditoria e estimativa exploratória de sobrevida global | Executada; requer verificações |
| 06 | Regressão de Cox e comparação de grupos de estágio | Exploratória |
| 06B | Diagnóstico de pressupostos e análises de sensibilidade | Em avaliação |
| 07 | Modelos preditivos e explicabilidade | Planejada |

## Sobrevida e modelagem estatística

A análise de sobrevida utiliza o estimador de **Kaplan–Meier**. A regressão de **riscos proporcionais de Cox** é empregada apenas para investigar associações exploratórias. O conjunto elegível e a definição de tempo/evento devem ser documentados em cada análise; não se deve equiparar automaticamente situação vital à disponibilidade de tempo válido para sobrevida.

**Verificações metodológicas prioritárias:**

- Coerência entre datas, tempos de acompanhamento, óbitos e censuras.
- Revisão de tempos nulos, negativos ou inconsistentes.
- Diagnóstico da hipótese de riscos proporcionais (resíduos de Schoenfeld).
- Avaliação da forma funcional de covariáveis contínuas e análises de sensibilidade.
- Controle de duplicidades e rastreabilidade das transformações.
- Interpretação cautelosa diante de grupos pequenos e intervalos de confiança amplos.

## Estrutura proposta do repositório

```text
TCGA-PAAD-Pancreatic-Cancer-AI-Research/
├── README.md
├── .gitignore
├── scripts/
│   ├── 02_exploracao_clinica.py
│   ├── 03_consolidacao_clinica.py
│   ├── 04_analise_exploratoria.py
│   ├── 05_analise_sobrevida.py
│   ├── 06_regressao_cox.py
│   └── 06b_diagnostico_cox.py
├── figures/
├── results/
└── docs/
```

*A estrutura é sugerida; os nomes dos arquivos devem ser ajustados aos scripts efetivamente versionados. Não inclua dados brutos, credenciais ou arquivos de acesso restrito sem revisar as condições de uso.*

## Reprodutibilidade e integridade científica

O repositório deve registrar versões de Python e dependências, comandos de execução, origem dos arquivos e decisões de limpeza e exclusão. Resultados numéricos devem ser recalculados a partir dos dados de entrada e comparados com as tabelas e figuras antes de serem considerados finais.

**Atenção:** o GDC pode impor regras específicas para acesso e redistribuição de determinados arquivos. Verifique as condições de cada modalidade e evite publicar dados individuais desnecessários.

## Limitações

Este piloto não estabelece capacidade de diagnóstico, predição clínica validada ou causalidade. A representatividade da coorte, a completude do seguimento, a qualidade da codificação de estágios, o pareamento entre modalidades e a validade dos pressupostos estatísticos exigem avaliação adicional. A etapa de inteligência artificial explicável permanece como perspectiva de pesquisa.

## Referências e recursos

- [National Cancer Institute — Genomic Data Commons](https://gdc.cancer.gov/)
- [TCGA-PAAD no portal GDC](https://portal.gdc.cancer.gov/projects/TCGA-PAAD)
- [Documentação da API do GDC](https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/)

## Autoria

**Bruno Bartolomeu**  
Pesquisa acadêmica em ciência de dados, inteligência artificial e saúde.  
[LinkedIn](https://www.linkedin.com/in/bruno-bartolomeu)

---

*Última atualização: novembro de 2026. Documentação de estudo piloto, sujeita a revisão metodológica.*
