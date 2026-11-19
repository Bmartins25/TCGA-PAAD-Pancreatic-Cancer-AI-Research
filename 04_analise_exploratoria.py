"""Piloto TCGA-PAAD | Etapa 04: análise exploratória clínica.

Uso: python 04_analise_exploratoria.py
Dependências: pip install pandas matplotlib numpy
Entrada: pacientes_tcga_paad_exploratorio.csv (mesma pasta do script)
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
AZUL = '#176B87'
VERDE = '#248B78'
VERMELHO = '#C45A61'
LARANJA = '#D18A36'
CINZA = '#607080'

def acabamento(ax, eixo='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=eixo, color='#E5EAF0', linewidth=.8)
    ax.tick_params(axis='both', length=0, pad=7)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#D6DEE6')

BASE = Path(__file__).resolve().parent
ENTRADA = BASE / 'pacientes_tcga_paad_exploratorio.csv'
SAIDA = BASE / 'resultados_exploratorios'
SAIDA.mkdir(exist_ok=True)

COLUNAS = {
    'id': 'cases.submitter_id',
    'idade': 'diagnoses.age_at_diagnosis',
    'estagio': 'diagnoses.ajcc_pathologic_stage',
    'vital': 'demographic.vital_status',
    'obito': 'demographic.days_to_death',
    'seguimento': 'diagnoses.days_to_last_follow_up',
}

def salvar(fig, nome):
    fig.savefig(SAIDA / nome, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Gráfico:', SAIDA / nome)

def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {ENTRADA}\nColoque o CSV na mesma pasta deste script.')
    df = pd.read_csv(ENTRADA, low_memory=False)
    faltantes = [c for c in COLUNAS.values() if c not in df.columns]
    if faltantes:
        raise ValueError('Colunas ausentes no CSV: ' + ', '.join(faltantes))
    df = df.replace(r'^\s*$', np.nan, regex=True)
    n = len(df)
    ids = df[COLUNAS['id']].astype('string')
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError('A base não possui um ID válido e único por paciente. Revise a consolidação antes de gerar gráficos.')
    print(f'Pacientes: {n} | IDs únicos: {ids.nunique()}')

    # No GDC, age_at_diagnosis costuma ser expressa em DIAS, não anos.
    idade_dias = pd.to_numeric(df[COLUNAS['idade']], errors='coerce')
    idade_anos = idade_dias / 365.25
    idade_anos = idade_anos.where(idade_anos.between(0, 120))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(idade_anos.dropna(), bins=12, color=AZUL, edgecolor='white', linewidth=1.4)
    ax.axvline(idade_anos.mean(), color=VERMELHO, linestyle='--', linewidth=2, label=f'Média: {idade_anos.mean():.1f} anos')
    ax.axvline(idade_anos.median(), color=VERDE, linestyle=':', linewidth=2.5, label=f'Mediana: {idade_anos.median():.1f} anos')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize=10)
    fig.subplots_adjust(top=.83)
    ax.set(title='Idade ao diagnóstico', xlabel='Idade (anos)', ylabel='Número de pacientes')
    acabamento(ax, 'y')
    salvar(fig, '01_idade_diagnostico.png')

    estagio = df[COLUNAS['estagio']].astype('string').str.strip()
    estagio = estagio.replace({'': pd.NA, '--': pd.NA, 'not reported': pd.NA, 'Not Reported': pd.NA})
    cont_estagio = estagio.fillna('Não informado').value_counts()
    ordem = ['Stage 0', 'Stage 0a', 'Stage IA', 'Stage IB', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IV']
    categorias = [x for x in ordem if x in cont_estagio.index] + sorted(x for x in cont_estagio.index if x not in ordem and x != 'Não informado')
    if 'Não informado' in cont_estagio.index: categorias.append('Não informado')
    valores = [int(cont_estagio[x]) for x in categorias]
    fig, ax = plt.subplots(figsize=(10, max(4.6, len(categorias)*.52 + 1)))
    cores = [CINZA if x == 'Não informado' else (VERMELHO if x == 'Stage IV' else VERDE) for x in categorias]
    barras = ax.barh(categorias[::-1], valores[::-1], color=cores[::-1], height=.66)
    ax.set_xlim(0, max(valores)*1.22)
    for barra, v in zip(barras, valores[::-1]):
        ax.text(v+max(valores)*.014, barra.get_y()+barra.get_height()/2, f'{v}  ({v/n:.1%})', va='center', fontsize=10, color='#253746')
    ax.set(title='Distribuição por estágio patológico', xlabel='Número de pacientes', ylabel='')
    acabamento(ax, 'x')
    fig.text(.12, .01, 'Categorias preservadas conforme o registro original do GDC; validar codificação clínica.', fontsize=8, color=CINZA)
    salvar(fig, '02_estadiamento.png')

    vital = df[COLUNAS['vital']].astype('string').str.strip().str.lower().map({'dead':'Falecido', 'alive':'Vivo'})
    vital = vital.fillna('Não informado')
    cont_vital = vital.value_counts().reindex(['Falecido','Vivo','Não informado'], fill_value=0)
    cont_vital = cont_vital[cont_vital > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(cont_vital.index, cont_vital.values, color=[{'Falecido': VERMELHO, 'Vivo': VERDE}.get(v, CINZA) for v in cont_vital.index], width=.52)
    for b, valor in zip(barras, cont_vital.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{valor} ({valor/n:.1%})', ha='center', va='bottom')
    ax.set(title='Situação vital na data de referência', ylabel='Número de pacientes', ylim=(0, max(cont_vital.values)*1.22))
    acabamento(ax, 'y')
    salvar(fig, '03_situacao_vital.png')

    # Percentual de ausência das variáveis aplicáveis a todos os pacientes.
    # A escala horizontal é percentual (não contagem de registros).
    rotulos = {'idade': 'Idade ao diagnóstico', 'estagio': 'Estágio patológico', 'vital': 'Situação vital'}
    idade_faltante = int(idade_anos.isna().sum())
    estagio_faltante = int(estagio.isna().sum())
    vital_faltante = int((vital == 'Não informado').sum())
    contagens = {'idade': idade_faltante, 'estagio': estagio_faltante, 'vital': vital_faltante}
    taxas = pd.Series({rotulos[k]: 100 * v / n for k, v in contagens.items()}).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    fig.subplots_adjust(left=.29, right=.95, bottom=.23, top=.82)
    cores = [LARANJA if v > 0 else VERDE for v in taxas.values]
    barras = ax.barh(taxas.index, taxas.values, color=cores, height=.43)
    # Escala curta e explicitamente identificada, para tornar 1,6% visível.
    limite = max(5.0, float(taxas.max()) * 1.4)
    ax.set_xlim(0, limite)
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
    for barra, valor, nome in zip(barras, taxas.values, taxas.index):
        chave = next(k for k, v in rotulos.items() if v == nome)
        ax.text(valor + .12, barra.get_y() + barra.get_height()/2,
                f'{valor:.1f}%  ({contagens[chave]}/{n})', va='center', fontsize=11, color='#253746')
    ax.set_title('Dados ausentes nas variáveis clínicas principais', pad=18)
    ax.set_xlabel('Percentual de registros ausentes', labelpad=10)
    ax.set_ylabel('')
    acabamento(ax, 'x')
    fig.text(.29, .075, 'Nota: dias até óbito e último acompanhamento são campos condicionais,', fontsize=9, color=CINZA)
    fig.text(.29, .045, 'avaliados separadamente conforme a situação vital.', fontsize=9, color=CINZA)
    salvar(fig, '04_dados_ausentes_corrigido.png')
    ausentes = pd.Series({COLUNAS['idade']: idade_faltante, COLUNAS['estagio']: estagio_faltante,
                          COLUNAS['vital']: vital_faltante,
                          COLUNAS['obito']: int(pd.to_numeric(df[COLUNAS['obito']], errors='coerce').isna().sum()),
                          COLUNAS['seguimento']: int(pd.to_numeric(df[COLUNAS['seguimento']], errors='coerce').isna().sum())})
    print('ATENÇÃO: gráfico 04 corrigido: somente idade, estágio e situação vital.')
    mortos = vital == 'Falecido'
    vivos = vital == 'Vivo'
    obito_num = pd.to_numeric(df[COLUNAS['obito']], errors='coerce')
    follow_num = pd.to_numeric(df[COLUNAS['seguimento']], errors='coerce')
    print('Dias até óbito ausentes entre falecidos:', int(obito_num[mortos].isna().sum()), '/', int(mortos.sum()))
    print('Dias até acompanhamento ausentes entre vivos:', int(follow_num[vivos].isna().sum()), '/', int(vivos.sum()))
    print('AVISO: a ausência desses campos não deve ser tratada automaticamente como falha de qualidade.')
    variaveis = [COLUNAS[k] for k in ('idade','estagio','vital','obito','seguimento')]

    # Ausência de dias até óbito em pacientes vivos pode ser esperada; não implica erro.
    resumo = pd.DataFrame([
        {'indicador':'pacientes_unicos', 'valor':n},
        {'indicador':'idade_media_anos', 'valor':round(idade_anos.mean(), 2)},
        {'indicador':'idade_mediana_anos', 'valor':round(idade_anos.median(), 2)},
        {'indicador':'idade_minima_anos', 'valor':round(idade_anos.min(), 2)},
        {'indicador':'idade_maxima_anos', 'valor':round(idade_anos.max(), 2)},
        {'indicador':'falecidos', 'valor':int((vital=='Falecido').sum())},
        {'indicador':'vivos', 'valor':int((vital=='Vivo').sum())},
        {'indicador':'situacao_vital_nao_informada', 'valor':int((vital=='Não informado').sum())},
    ] + [{'indicador':'ausentes_'+re.sub(r'[^a-z0-9]+','_',c.lower()), 'valor':int(ausentes[c])} for c in variaveis])
    resumo.to_csv(SAIDA / 'resumo_estatistico.csv', index=False, encoding='utf-8-sig')
    cont_estagio.rename_axis('estagio').reset_index(name='pacientes').to_csv(SAIDA / 'frequencia_estagios.csv', index=False, encoding='utf-8-sig')
    print('\nResumo:\n', resumo.to_string(index=False))
    print('\nATENÇÃO: dados exploratórios; não representam análise de sobrevida nem validação de modelo de IA.')
    print('Verifique a seleção de diagnósticos e acompanhamentos na consolidação antes da modelagem.')
    print('Resultados salvos em:', SAIDA)

if __name__ == '__main__':
    main()
