TCGA-PAAD | Pancreatic Cancer AI Research
Estudo piloto de análise clínica e de sobrevida em câncer de pâncreas, com perspectiva de desenvolvimento de modelos de inteligência artificial explicável.
Visão geral
Este repositório documenta um estudo piloto computacional baseado em dados secundários do The Cancer Genome Atlas – Pancreatic Adenocarcinoma (TCGA-PAAD), disponibilizados pelo Genomic Data Commons (GDC) do National Cancer Institute.

O trabalho integra a preparação de dados clínicos, a análise exploratória de pacientes e a avaliação preliminar de modelos de sobrevida. Seu propósito é apoiar o amadurecimento metodológico de uma proposta de doutorado voltada à predição de desfechos clínicos no câncer de pâncreas por meio da integração de dados clínicos e multiômicos e de técnicas de inteligência artificial explicável.
Status: estudo piloto exploratório. Os resultados apresentados não representam validação clínica, ferramenta de diagnóstico ou modelo pronto para uso assistencial.

Objetivos
- Organizar e auditar dados clínicos secundários do TCGA-PAAD.
- Caracterizar a coorte quanto a idade, estadiamento, situação vital e completude dos registros.
- Investigar a sobrevida global por métodos descritivos e estatísticos.
- Explorar associações entre características clínicas e risco de óbito por regressão de Cox.
- Avaliar pressupostos, sensibilidade das estimativas e limitações metodológicas.
- Preparar uma base reprodutível para etapas futuras de aprendizado de máquina e integração multiômica.
Fonte dos dados
- Programa: The Cancer Genome Atlas (TCGA)
- Projeto: TCGA-PAAD
- Portal: NCI Genomic Data Commons
- API: GDC API
A disponibilidade de arquivos, permissões e regras de uso deve ser verificada diretamente no GDC. Os dados brutos não são necessariamente redistribuídos neste repositório.
Fluxo analítico
Etapa	Conteúdo	Situação
01	Obtenção e organização dos arquivos clínicos e de RNA-seq	Em andamento
02	Inspeção e exploração das tabelas clínicas	Executada
03	Consolidação de registros por paciente	Executada
04	Análise exploratória e visualização	Executada
05	Auditoria e análise exploratória de sobrevida global	Executada
06	Regressão de Cox e comparação de grupos de estágio	Executada, com pressupostos pendentes de avaliação completa
06B	Diagnósticos de resíduos e análise de sensibilidade	Etapa de diagnóstico
07	Modelos de aprendizado de máquina para sobrevida	Planejada


As designações correspondem às etapas do estudo; a execução de uma etapa não implica validação definitiva de seus resultados.
Coorte exploratória
Na consolidação clínica inicial, foram identificados 185 pacientes únicos. Na análise exploratória de sobrevida, foram considerados 184 registros com tempo e desfecho utilizáveis (100 eventos e 84 censuras). Uma análise subsequente de regressão de Cox utilizou 179 registros elegíveis, após exclusões específicas do modelo.

Atenção: os tamanhos amostrais variam de acordo com os critérios de elegibilidade de cada análise. Antes de qualquer publicação, é necessário auditar os tempos de seguimento, a codificação dos eventos, as categorias de estágio e os critérios de exclusão. As estimativas não devem ser interpretadas como desempenho preditivo de IA.
Métodos
Análise descritiva: estatísticas de tendência central, distribuição etária, frequências de estadiamento e completude de variáveis.

Sobrevida: estimador de Kaplan–Meier e identificação de censuras, com conferência da origem temporal dos campos clínicos.

Regressão de Cox: estimação exploratória de hazard ratios e intervalos de confiança; investigação de resíduos de Schoenfeld e Martingale e análises de sensibilidade. A verificação formal de riscos proporcionais e da forma funcional das covariáveis é necessária antes de interpretações confirmatórias.

Etapas futuras: validação adequada por paciente, modelos de sobrevivência baseados em aprendizado de máquina, métricas de discriminação e calibração e, conforme disponibilidade e compatibilidade dos dados, integração de modalidades ômicas e explicabilidade.
Tecnologias
- Python
- pandas e NumPy
- Matplotlib
- statsmodels
- Ferramentas do GDC para aquisição de dados
As versões efetivamente utilizadas devem ser registradas em um arquivo requirements.txt ou ambiente reproduzível.
Reprodutibilidade
1. Consulte as condições de acesso e obtenha os arquivos correspondentes no GDC.
2. Configure um ambiente Python isolado.
3. Instale as dependências declaradas pelo projeto.
4. Execute os scripts em ordem, respeitando as entradas e saídas documentadas.
5. Confira os relatórios de auditoria e os critérios de elegibilidade antes de interpretar os gráficos.
Boas práticas: não versionar credenciais, tokens, arquivos de acesso controlado, grandes downloads brutos nem dados pessoais. Recomenda-se armazenar os dados em diretório local ignorado pelo Git e registrar apenas metadados, scripts e resultados cuja divulgação seja permitida.
Limitações
- Coorte relativamente pequena e sujeita a seleção e dados ausentes.
- Possível heterogeneidade clínica e de codificação do estadiamento.
- Necessidade de verificar consistência entre datas, tempos de seguimento e eventos.
- Ausência, nesta fase, de validação externa e avaliação de utilidade clínica.
- Dados clínicos e ômicos não necessariamente pareados por paciente ou comparáveis entre fontes.
Uso científico e ética
Este repositório destina-se exclusivamente à pesquisa e ao desenvolvimento metodológico. Nenhuma análise deve ser utilizada para tomada de decisão clínica individual. A utilização e a redistribuição dos dados devem observar as políticas aplicáveis do GDC e das instituições envolvidas.
Autoria
Bruno Bartolomeu
Pesquisa em Ciência de Dados, Inteligência Artificial Responsável e Saúde Digital.
Referências e reconhecimento
- National Cancer Institute. Genomic Data Commons (GDC): https://gdc.cancer.gov/
- The Cancer Genome Atlas Research Network: https://www.cancer.gov/ccg/research/genome-sequencing/tcga
