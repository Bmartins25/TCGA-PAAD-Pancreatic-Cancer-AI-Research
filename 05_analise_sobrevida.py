# -*- coding: utf-8 -*-
"""Piloto TCGA-PAAD | Script 05: auditoria e Kaplan-Meier exploratório.
Dependências: pandas, numpy, matplotlib. Não altera a base original.
Executar: python 05_analise_sobrevida.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ENTRADA = BASE / 'pacientes_tcga_paad_exploratorio.csv'
SAIDA = BASE / 'resultados_sobrevida'
SAIDA.mkdir(exist_ok=True)

COL_ID = 'cases.submitter_id'
COL_STATUS = 'demographic.vital_status'
COL_DEATH = 'demographic.days_to_death'
COL_FOLLOW = 'diagnoses.days_to_last_follow_up'
COL_AGE = 'diagnoses.age_at_diagnosis'
COL_STAGE = 'diagnoses.ajcc_pathologic_stage'


def km_table(tempos, eventos):
    """Estimador Kaplan-Meier; censuras no mesmo instante após eventos."""
    dados = pd.DataFrame({'tempo': tempos, 'evento': eventos}).sort_values('tempo')
    risco = len(dados)
    surv = 1.0
    rows = [{'tempo_dias': 0., 'em_risco': risco, 'eventos': 0, 'censuras': 0, 'sobrevida': 1.}]
    for t, grupo in dados.groupby('tempo', sort=True):
        d = int(grupo['evento'].sum())
        c = len(grupo) - d
        if d:
            surv *= (1 - d / risco)
        rows.append({'tempo_dias': float(t), 'em_risco': risco,
                     'eventos': d, 'censuras': c, 'sobrevida': surv})
        risco -= len(grupo)
    return pd.DataFrame(rows)


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {ENTRADA}')
    df = pd.read_csv(ENTRADA, dtype={COL_ID: 'string', COL_STATUS: 'string', COL_STAGE: 'string'})
    required = [COL_ID, COL_STATUS, COL_DEATH, COL_FOLLOW, COL_AGE, COL_STAGE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Colunas ausentes: {missing}')
    if df[COL_ID].isna().any() or df[COL_ID].duplicated().any():
        raise ValueError('IDs ausentes ou duplicados: revisar consolidação antes de continuar.')
    for col in [COL_DEATH, COL_FOLLOW, COL_AGE]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    status = df[COL_STATUS].str.strip().str.lower()
    df['evento_obito'] = status.map({'dead': 1, 'alive': 0})
    df['tempo_sobrevida_dias'] = np.where(df['evento_obito'].eq(1), df[COL_DEATH], df[COL_FOLLOW])
    df['idade_anos'] = df[COL_AGE] / 365.25
    flags = []
    for _, r in df.iterrows():
        problemas = []
        if pd.isna(r['evento_obito']): problemas.append('situacao_vital_desconhecida')
        t = r['tempo_sobrevida_dias']
        if pd.isna(t): problemas.append('tempo_sobrevida_ausente')
        elif t < 0: problemas.append('tempo_sobrevida_negativo')
        elif t == 0: problemas.append('tempo_sobrevida_zero')
        if pd.isna(r[COL_AGE]) or r[COL_AGE] <= 0: problemas.append('idade_invalida')
        if pd.isna(r[COL_STAGE]): problemas.append('estagio_ausente')
        elif str(r[COL_STAGE]).strip().lower() in ('stage 0', 'stage 0a'):
            problemas.append('estagio_atipico_revisar_gdc')
        if r['evento_obito'] == 0 and pd.notna(r[COL_DEATH]):
            problemas.append('vivo_com_dias_ate_obito')
        if r['evento_obito'] == 1 and pd.notna(r[COL_FOLLOW]) and pd.notna(r[COL_DEATH]):
            if r[COL_FOLLOW] > r[COL_DEATH]:
                problemas.append('acompanhamento_posterior_obito')
        flags.append('; '.join(problemas))
    df['alertas_auditoria'] = flags
    df['elegivel_km'] = df['evento_obito'].notna() & df['tempo_sobrevida_dias'].gt(0)
    df.to_csv(SAIDA / 'base_sobrevida_auditada.csv', index=False, encoding='utf-8-sig')
    df.loc[df['alertas_auditoria'].ne(''), [COL_ID, 'alertas_auditoria',
        'tempo_sobrevida_dias', 'evento_obito']].to_csv(
        SAIDA / 'pendencias_auditoria.csv', index=False, encoding='utf-8-sig')
    valido = df.loc[df['elegivel_km']].copy()
    if valido.empty: raise ValueError('Nenhum registro elegível para Kaplan-Meier.')
    tab = km_table(valido['tempo_sobrevida_dias'], valido['evento_obito'].astype(int))
    tab.to_csv(SAIDA / 'tabela_kaplan_meier.csv', index=False, encoding='utf-8-sig')
    med = tab.loc[tab['sobrevida'] <= .5, 'tempo_dias']
    mediana = float(med.iloc[0]) if len(med) else None

    # Visual refinado: eixo em meses, mediana e tabela de risco alinhada.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    azul, laranja, cinza = '#176b80', '#c77b46', '#516174'
    fig = plt.figure(figsize=(12.8, 7.6), facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.05], hspace=.12,
                         left=.105, right=.955, top=.82, bottom=.17)
    ax = fig.add_subplot(gs[0]); riskax = fig.add_subplot(gs[1], sharex=ax)
    x = tab['tempo_dias'].to_numpy(dtype=float) / 30.4375
    y = tab['sobrevida'].to_numpy(dtype=float)
    ax.step(x, y, where='post', color=azul, lw=2.6, label='Kaplan–Meier')
    cens = tab.loc[tab['censuras'] > 0]
    ax.scatter(cens['tempo_dias'].to_numpy(dtype=float) / 30.4375,
               cens['sobrevida'].to_numpy(dtype=float), marker='|', s=48,
               linewidths=1.15, color=laranja, zorder=3, label='Censura')
    if mediana is not None:
        med_m = mediana / 30.4375
        ax.hlines(.5, 0, med_m, color=cinza, lw=1.15, ls=(0,(4,4)), alpha=.7)
        ax.vlines(med_m, 0, .5, color=cinza, lw=1.15, ls=(0,(4,4)), alpha=.7)
        ax.annotate(f'Mediana: {med_m:.1f} meses', xy=(med_m,.5),
                    xytext=(med_m+5,.60), fontsize=10.5, color='#243746',
                    arrowprops={'arrowstyle':'-', 'color':cinza, 'lw':1})
    max_months = float(valido['tempo_sobrevida_dias'].max()) / 30.4375
    ticks = np.arange(0, 91, 12)
    ticks = ticks[ticks <= max_months + 1]
    if len(ticks) < 2: ticks = np.linspace(0,max_months,5)
    ax.set_xlim(0, max_months + 2)
    ax.set_ylim(0,1.04)
    ax.set_yticks(np.arange(0,1.01,.2))
    ax.set_ylabel('Probabilidade de sobrevida', fontsize=11, labelpad=12)
    ax.grid(axis='y', color='#e6ecf0', linewidth=.9)
    ax.spines['left'].set_color('#bac7d0')
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', labelbottom=False, bottom=False)
    ax.tick_params(axis='y', colors=cinza)
    ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=10)
    fig.suptitle('Sobrevida global | TCGA-PAAD', fontsize=20, weight='bold',
                 color='#18354a', y=.955)
    fig.text(.105,.885, f'Análise exploratória  •  n = {len(valido)}  •  '
             f'óbitos = {int(valido["evento_obito"].sum())}  •  '
             f'censuras = {int((valido["evento_obito"] == 0).sum())}',
             color=cinza, fontsize=11)
    at_risk = [int((valido['tempo_sobrevida_dias'] / 30.4375 >= t).sum()) for t in ticks]
    riskax.set_xlim(ax.get_xlim()); riskax.set_ylim(0,1)
    riskax.set_xticks(ticks)
    riskax.set_xticklabels([f'{t:g}' for t in ticks], fontsize=10, color=cinza)
    riskax.set_yticks([])
    riskax.set_xlabel('Tempo desde o diagnóstico (meses)', fontsize=11, labelpad=8)
    riskax.text(0, .92, 'Pacientes em risco', fontsize=10.5, weight='bold',
                color='#18354a', transform=riskax.transAxes, va='top')
    for t,n in zip(ticks,at_risk):
        riskax.text(t,.42,str(n),ha='center',va='center',fontsize=10,color='#18354a')
    for spine in riskax.spines.values(): spine.set_visible(False)
    riskax.tick_params(axis='x', length=0)
    fig.text(.105,.055, 'Fonte: TCGA-PAAD / GDC  •  Elaboração própria (2026)  •  '
             'Estimativa exploratória; requer validação clínica e temporal.',
             color='#64748b', fontsize=9)
    destino = SAIDA / '05_kaplan_meier_refinado.png'
    fig.savefig(destino, dpi=240, facecolor='white')
    plt.close(fig)

    texto = [
        'PILOTO TCGA-PAAD — AUDITORIA E SOBREVIDA EXPLORATÓRIA',
        f'Arquivo: {ENTRADA.name}',
        f'Pacientes únicos: {len(df)}',
        f'Óbitos: {int(df["evento_obito"].eq(1).sum())}',
        f'Vivos: {int(df["evento_obito"].eq(0).sum())}',
        f'Elegíveis KM (tempo > 0): {len(valido)}',
        f'Não elegíveis KM: {int((~df["elegivel_km"]).sum())}',
        f'Registros com algum alerta: {int(df["alertas_auditoria"].ne("").sum())}',
        f'Mediana KM (dias): {mediana if mediana is not None else "não atingida"}',
        '',
        'REGRAS: falecido -> days_to_death, evento=1; vivo -> days_to_last_follow_up, evento=0.',
        'Tempos zero/negativos/ausentes não entram na curva principal; permanecem auditados.',
        'Campos condicionais ausentes não são automaticamente imputados.',
        'Estágios 0/0a requerem revisão da codificação original.',
        'A data de origem dos tempos e o alinhamento entre arquivos GDC devem ser verificados.',
        'Curva descritiva sem ajuste de confundidores; não representa validação de IA.',
        'Sem intervalos de confiança ou inferência causal nesta versão.',
    ]
    (SAIDA / 'relatorio_auditoria.txt').write_text('\n'.join(texto), encoding='utf-8')
    print('\n'.join(texto))
    print(f'\nArquivos gerados em: {SAIDA}')

if __name__ == '__main__':
    main()
