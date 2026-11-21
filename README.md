# TCGA-PAAD | Pancreatic Cancer AI Research

**Estudo piloto de análise clínica, sobrevida e inteligência artificial interpretável em câncer de pâncreas**

**Orientação: Prof. Jerônimo Ruiz - Instituto René Rachou - Fiocruz Minas**

![Status](https://img.shields.io/badge/status-pipeline%20explorat%C3%B3rio%20conclu%C3%ADdo-0f766e) ![Dataset](https://img.shields.io/badge/dados-TCGA--PAAD-2563eb) ![Python](https://img.shields.io/badge/linguagem-Python-3776ab) ![Validação](https://img.shields.io/badge/valida%C3%A7%C3%A3o-externa%20pendente-d97706)

> **Escopo científico:** estudo computacional retrospectivo e exploratório com dados secundários do TCGA-PAAD, acessados por meio do Genomic Data Commons (GDC). O pipeline de exploração, integração, modelagem, interpretabilidade, robustez e consolidação documental foi executado. **Não se trata de modelo clinicamente validado, biomarcador validado ou evidência causal.** A auditoria global de integridade SHA-256 (13B) permanece pendente.

## Visão geral

O projeto investiga dados clínicos e de expressão gênica (RNA-seq) em adenocarcinoma pancreático, com foco na predição exploratória de sobrevida global e na comparação de modelos clínico, transcriptômico e integrado. Constitui uma base metodológica para pesquisas futuras em inteligência artificial explicável e integração multiômica.

## Objetivos

- Organizar e auditar registros clínicos e arquivos de RNA-seq do TCGA-PAAD.
- Caracterizar idade, estágio, situação vital, tempos de seguimento e censura.
- Explorar curvas de Kaplan–Meier e regressão de riscos proporcionais de Cox, com diagnósticos e análises de sensibilidade.
- Realizar controle de qualidade de expressão, pareamento por paciente e integração clínica–transcriptômica.
- Comparar modelos de sobrevida clínico, transcriptômico e integrado sob validação cruzada repetida.
- Descrever contribuições de covariáveis e estabilidade da seleção gênica, sem atribuir causalidade.
- Consolidar resultados e documentar limitações e pendências de reprodutibilidade.

## Fonte dos dados e desenho

| Item | Descrição |
|:--|:--|
| Programa | The Cancer Genome Atlas (TCGA) |
| Projeto | TCGA-PAAD — Pancreatic Adenocarcinoma |
| Portal | [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/projects/TCGA-PAAD) |
| Modalidades analisadas | Dados clínicos e contagens de RNA-seq |
| Desenho | Observacional retrospectivo, secundário e exploratório |
| Unidade de pareamento | Identificador do paciente (não por data de coleta) |
| Desfecho exploratório | Sobrevida global, com evento e censura conforme dados disponíveis |
| Ambiente utilizado | Python 3.9; execução local em Windows/VS Code |

### População e rastreabilidade das amostras

| Marco do processamento | Pacientes |
|:--|--:|
| Coorte clínica consolidada | **185** |
| Pacientes com matriz RNA-seq na integração | **178** |
| Pacientes pareados com desfecho válido | **177** |
| Pacientes elegíveis para a modelagem final | **172** |

A redução entre etapas decorre dos critérios e disponibilidades específicos de cada análise; esses totais **não devem ser confundidos**. Na caracterização clínica preliminar, registraram-se idade média de 65,42 anos, mediana de 65,83 anos, 100 falecidos, 85 vivos e três registros sem estágio patológico. Esses indicadores exigem conferência de definições e proveniência antes de publicação.

## Fluxo analítico e situação das etapas

| Etapa | Conteúdo | Situação documental |
|:--|:--|:--|
| 01 | Validação inicial e organização dos dados | Executada no piloto; aquisição e completude devem ser auditadas |
| 02 | Exploração clínica | Executada |
| 03 | Consolidação de pacientes | Executada |
| 04 | Análise exploratória e visualizações | Executada |
| 05 | Sobrevida global, Kaplan–Meier e censura | Executada em caráter exploratório |
| 06 | Regressão de Cox | Executada em caráter exploratório |
| 06B | Diagnósticos de Cox | Executada; pressupostos requerem revisão científica |
| 06C | Validação formal exploratória de Cox | Executada; não equivale a validação externa |
| 07 | Modelagem inicial de sobrevida | Executada |
| 08 | Validação e métricas por dobra | Executada; registros preservados |
| 09 / 09B | Mapeamento e processamento de RNA-seq | Executadas |
| 09C | Controle de qualidade e exploração da expressão | Executada |
| 09D | Auditoria de integração clínica–transcriptômica | Executada; exportação de matriz bruta documentada |
| 10 | Comparação dos três modelos em seis dobras | Executada |
| 10B | Diferenças pareadas e estabilidade de genes | Executada; análise descritiva |
| 11 | Interpretabilidade por decomposição do log-risco | Executada; não é SHAP |
| 12 | Robustez descritiva e subgrupos | Executada; não é teste de superioridade |
| 13 | Consolidação e auditoria documental | Executada: 14 verificações, zero alertas no escopo examinado |
| 13B | Auditoria global de integridade SHA-256 | **Não concluída**; interrompida durante leitura de arquivos |
| 14 | Relatório científico documental | Executada; pendências inicialmente identificadas |
| 14B | Recuperação de métricas auditadas da etapa 13 | Executada |
| 14C | Conferência do CSV da etapa 10 contra a etapa 13 | **Executada: 3/3 modelos, diferenças zero, 0 alertas documentais** |

> Os estados acima descrevem as execuções e saídas observadas no piloto local. Não substituem uma revisão independente de código, entradas, critérios de elegibilidade e integridade de todos os arquivos.

## Processamento transcriptômico

- Inspeção da profundidade das bibliotecas e da complexidade observada por amostra.
- Matriz inicial com **60.616 genes/identificadores**; filtragem exploratória reteve **18.551** e removeu **42.065**.
- Visualização da distribuição após transformação `log2(CPM + 1)` e análise de componentes principais (PCA) para controle de qualidade.
- Pareamento entre dados clínicos e expressão por identificador de paciente.
- A matriz bruta exportada na etapa 09D é destinada à integração auditável. Transformações e seleção de variáveis para predição devem ser ajustadas **somente nos dados de treino de cada dobra**, evitando vazamento de informação.

A PCA exploratória foi ajustada na coorte para controle de qualidade e **não** deve ser reutilizada como transformação pré-ajustada na validação preditiva.

## Modelagem e validação

Foram comparados três conjuntos de covariáveis: **clínico**, **transcriptômico** e **integrado**. A modelagem exploratória utilizou seis dobras registradas, com preparação de atributos e seleção gênica dentro do treino, conforme descrito nas saídas das etapas 10–12. A seleção por variância não comprova relevância biológica.

### Desempenho discriminativo

| Modelo | C-index médio | DP descritivo entre dobras | Mediana |
|:--|--:|--:|--:|
| Clínico | **0,550866** | 0,035477 | 0,566416 |
| Integrado | **0,538341** | 0,033291 | 0,542032 |
| Transcriptômico | **0,486051** | 0,030700 | 0,485726 |

*Os C-index médios foram conferidos entre `resultados_modelagem_10/10_resumo_modelos.csv` e `resultados_robustez_12/resultados_consolidacao_13/13_resumo.json` na etapa 14C: três correspondências, diferenças exibidas de 0,000000 e nenhum alerta documental remanescente.*

### Comparações pareadas descritivas

| Comparação | Δ médio de C-index | Dobras com Δ positivo | Dobras com Δ negativo |
|:--|--:|--:|--:|
| Integrado − clínico | −0,012525 | 1 | 5 |
| Integrado − transcriptômico | +0,052290 | 5 | 1 |
| Transcriptômico − clínico | −0,064815 | 1 | 5 |

O modelo clínico apresentou a maior discriminação média **neste experimento**. Não há demonstração de superioridade estatística, benefício clínico ou generalização externa. As dobras de validação cruzada repetida não são observações independentes; os desvios-padrão apresentados são descritivos, não intervalos de confiança.

## Interpretabilidade e estabilidade de atributos

A etapa 11 utilizou decomposição exata do preditor linear do modelo de Cox, calculando contribuições do tipo `β × (x − média do treino)` para o log-risco. **Não foi utilizado SHAP**. Os resultados descrevem a influência matemática das covariáveis no modelo ajustado, sem estabelecer efeito causal ou biomarcador validado.

Na avaliação exploratória de estabilidade gênica da etapa 10B, foram observados **22 genes distintos**, dos quais **18 apareceram em todas as seis dobras**; o índice de Jaccard médio descritivo foi aproximadamente **0,9394**. Identificadores Ensembl foram mantidos sem anotação biológica externa verificada.

## Robustez e subgrupos

A etapa 12 examinou discriminação por dobra, diferenças pareadas e subgrupos de idade e estágio. As análises estratificadas por evento isolado não produziram C-index calculável; estratos com poucos eventos ou amostras não sustentam conclusões robustas. A ausência de controle completo para lote, pureza tumoral, origem temporal e qualidade biológica limita a interpretação.

## Consolidação e auditoria documental

- **Etapa 13:** 12 arquivos inventariados no diretório examinado, 14 verificações documentais e zero alertas.
- **Etapa 14:** inventário documental mais amplo de 83 arquivos; cinco pendências iniciais de localização/extração.
- **Etapa 14B:** C-index recuperados do resumo auditado da etapa 13, com backup do relatório.
- **Etapa 14C:** leitura do resumo CSV da etapa 10, confronto com a etapa 13 e atualização do relatório; **zero alertas restantes no escopo da consolidação documental**.
- **Etapa 13B:** a tentativa de hash global foi interrompida (`KeyboardInterrupt`); **não há certificação global de integridade SHA-256**.

A ausência de alertas na etapa 14C **não** significa ausência de limitações metodológicas nem conclusão da auditoria global 13B.

## Organização sugerida do repositório

```text
TCGA-PAAD-Pancreatic-Cancer-AI-Research/
├── README.md
├── .gitignore
├── requirements.txt                 # gerar a partir do ambiente revisado
├── scripts/                         # scripts efetivamente versionados
│   ├── 01_validacao_dados.py
│   ├── 02_exploracao_clinica.py
│   ├── 03_consolidacao_pacientes.py
│   ├── ...                           # etapas intermediárias 04–12
│   ├── 13_consolidacao_cientifica_auditoria.py
│   ├── 13b_auditoria_global.py
│   ├── 14_relatorio_cientifico_final.py
│   ├── 14b_corrigir_consolidacao.py
│   └── 14c_verificacao_final_modelagem.py
├── figures/                         # figuras revisadas para publicação
├── results/                         # tabelas e resumos sem dados restritos
│   ├── modelagem_10/
│   ├── robustez_12/
│   ├── consolidacao_13/
│   └── relatorio_14/
└── docs/                            # protocolo, dicionário e limitações
```

**Nota:** esta árvore é uma **proposta de organização para o GitHub**, não uma representação literal dos diretórios locais. Os arquivos intermediários devem ser incluídos com seus nomes reais após conferência. Não renomeie scripts sem ajustar caminhos e dependências.

### Arquivos de evidência da conferência final

```text
resultados_modelagem_10/10_resumo_modelos.csv
resultados_robustez_12/12_resumo.json
resultados_robustez_12/resultados_consolidacao_13/13_resumo.json
resultados_relatorio_14/14_resumo.json
resultados_relatorio_14/14_metricas.csv
```

## Reprodutibilidade e integridade científica

Para uma reprodução independente, ainda é necessário publicar um ambiente de dependências validado, documentar versões de dados/arquivos GDC e manifestos, explicitar critérios de inclusão e exclusão, disponibilizar comandos de execução e revisar a integridade das entradas. A execução local utilizou Python 3.9, mas a simples indicação da versão não garante reprodutibilidade.

**Boas práticas de compartilhamento:** não publicar credenciais, tokens, dados individuais sensíveis ou arquivos sujeitos a restrições de redistribuição; revisar os termos do GDC e preferir scripts, metadados permitidos e saídas agregadas. Evitar caminhos absolutos com nomes de usuários em arquivos públicos.

## Limitações e próximos passos científicos

1. Auditoria global 13B de integridade ainda não concluída.
2. Ausência de validação externa e de avaliação prospectiva.
3. Necessidade de revisar temporalidade, definição do desfecho, censura e elegibilidade.
4. Falta de harmonização completa de lote, pureza tumoral e potenciais confundidores.
5. Falta de anotação e validação biológica independente dos genes selecionados.
6. Desempenho discriminativo modesto e ausência de teste conclusivo de superioridade entre modelos.
7. Ausência de demonstração de utilidade clínica, causalidade ou transportabilidade.
8. Para estudos futuros: avaliação de calibração, métricas dependentes do tempo, validação externa e integração de outras modalidades quando houver pareamento e qualidade adequados.

## Ética e uso responsável

Este piloto emprega dados secundários para pesquisa metodológica. Nenhuma saída deve ser usada para orientar diagnóstico ou tratamento. O uso e a redistribuição dos arquivos devem observar as regras aplicáveis do GDC, proteção de dados e práticas de pesquisa responsável.

## Referências e recursos

- [National Cancer Institute — Genomic Data Commons](https://gdc.cancer.gov/)
- [Projeto TCGA-PAAD no GDC](https://portal.gdc.cancer.gov/projects/TCGA-PAAD)
- [Documentação da API do GDC](https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/)
- [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)

## Autoria

**Bruno Bartolomeu**  
Pesquisa acadêmica em ciência de dados, inteligência artificial e saúde.  
[LinkedIn](https://www.linkedin.com/in/bruno-bartolomeu)

---

*Última atualização: novembro de 2026. Pipeline exploratório documentado até a etapa 14C; auditoria global 13B e validação científica externa pendentes.*
