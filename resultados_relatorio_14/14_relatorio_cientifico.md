# TCGA-PAAD — Relatório científico consolidado (Etapa 14)

**Gerado em:** 2026-11-18T21:35:48.159083-03:00

**Escopo:** auditoria documental e síntese exploratória de dados clínicos e RNA-seq.

## 1. Objetivo

Consolidar resultados das etapas 09 a 13 com rastreabilidade documental, sem reprocessar RNA-seq nem retreinar modelos.

## 2. Coorte e integração

| etapa | indicador | valor |
| --- | --- | --- |
| 09d | pacientes_clinicos/pacientes_clinica | 185 |
| 09b | pacientes_com_matriz/pacientes_com_rna_tumor_primario | 178 |
| 09d | pacientes_pareados_desfecho_valido/pacientes_pareados | 177 |
| 11 | pacientes_elegiveis | 172 |


As contagens acima são exibidas somente quando identificadas nos resumos disponíveis; devem ser conferidas com as definições de coorte e endpoint.

## 3. Desempenho preditivo exploratório

| modelo | c_index_medio | fonte |
| --- | --- | --- |
| clinico | Não disponível | Não verificada |
| integrado | Não disponível | Não verificada |
| transcriptomico | Não disponível | Não verificada |


C-index próximo de 0,5 indica discriminação limitada. Diferenças descritivas entre modelos não demonstram superioridade estatística.

Os valores são omitidos quando não puderam ser extraídos de uma fonte estruturada conhecida.

## 4. Interpretabilidade

A etapa 11 avaliou contribuições ao log-risco de um modelo de Cox; tais contribuições não são efeitos causais nem validação de biomarcadores.

## 5. Robustez

As comparações entre dobras repetidas são dependentes. Não interpretar desvios-padrão entre dobras como intervalos de confiança independentes.

Análises por subgrupos com poucos eventos ou dobras não avaliáveis não permitem conclusões estáveis.

## 6. Limitações

- Coorte retrospectiva observacional e amostra restrita ao TCGA-PAAD.

- Necessário verificar temporalidade, censura, origem dos dados, efeitos de lote e qualidade biológica.

- A etapa 09C foi exploratória; filtros, escalonamento e seleção para predição devem ocorrer exclusivamente dentro de cada dobra de treino.

- Não foram demonstradas validação externa, utilidade clínica, causalidade ou generalização.

- A auditoria SHA-256 global 13B foi interrompida e não é considerada concluída.

## 7. Rastreabilidade e pendências

Foram inventariados 83 arquivos diretos nas pastas encontradas, sem leitura de matrizes brutas.

### Alertas e verificações pendentes

- Resumo JSON não identificado para etapa 10; dados não serão presumidos.
- Resumo JSON da etapa 13 não localizado por nome conhecido; confira o conteúdo da pasta.
- C-index clinico: não extraído de resumo JSON conhecido; valor do terminal não foi adotado automaticamente.
- C-index integrado: não extraído de resumo JSON conhecido; valor do terminal não foi adotado automaticamente.
- C-index transcriptomico: não extraído de resumo JSON conhecido; valor do terminal não foi adotado automaticamente.

## 8. Conclusão

Este documento consolida evidências exploratórias disponíveis. Não constitui validação clínica, biológica ou externa dos modelos.

Recomenda-se revisar as fontes, completar a avaliação de calibração e validação externa quando houver coorte independente apropriada.
