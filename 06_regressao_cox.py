# -*- coding: utf-8 -*-
"""Script 06 | Cox exploratorio TCGA-PAAD. Python >=3.9.
Instalar: python -m pip install pandas numpy matplotlib scipy statsmodels
Executar na pasta que contem pacientes_tcga_paad_exploratorio.csv:
    python 06_regressao_cox.py
Nao modifica arquivos originais. Associacoes, nao predicao validada.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.duration.hazard_regression import PHReg

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'pacientes_tcga_paad_exploratorio.csv'
OUT = ROOT / 'resultados_cox'
OUT.mkdir(parents=True, exist_ok=True)
C = {'id':'cases.submitter_id', 'status':'demographic.vital_status',
     'death':'demographic.days_to_death', 'follow':'diagnoses.days_to_last_follow_up',
     'age':'diagnoses.age_at_diagnosis', 'stage':'diagnoses.ajcc_pathologic_stage'}
NAVY='#18364A'; TEAL='#168A82'; ORANGE='#D08A3C'; RED='#BE6470'; MUTED='#637487'; GRID='#E5EBEF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.facecolor':'white'})

def save(fig, name):
    fig.savefig(OUT/name, dpi=260, facecolor='white', bbox_inches='tight', pad_inches=.22)
    plt.close(fig)
    print('Gráfico:', OUT/name)

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Coloque o CSV original ao lado do script: {SOURCE}')
    d=pd.read_csv(SOURCE)
    missing=[v for v in C.values() if v not in d.columns]
    if missing: raise ValueError(f'Colunas ausentes: {missing}')
    if d[C['id']].isna().any() or d[C['id']].duplicated().any():
        raise ValueError('IDs ausentes ou duplicados: interrompendo modelagem')
    for k in ('death','follow','age'):
        d[C[k]]=pd.to_numeric(d[C[k]],errors='coerce')
    d['evento']=d[C['status']].astype('string').str.strip().str.lower().map({'dead':1,'alive':0})
    d['tempo_dias']=np.where(d['evento']==1,d[C['death']],d[C['follow']])
    d['idade_anos']=d[C['age']]/365.25
    stage=d[C['stage']].astype('string').str.strip().str.upper()
    # I/II/III-IV: III-IV contem poucos casos; apenas analise exploratoria.
    mapping={'STAGE IA':'I','STAGE IB':'I','STAGE IIA':'II','STAGE IIB':'II',
             'STAGE III':'III-IV','STAGE IV':'III-IV'}
    d['grupo_estagio']=stage.map(mapping)
    issues=[]
    for _,r in d.iterrows():
        flags=[]
        if pd.isna(r['evento']): flags.append('status_desconhecido')
        if pd.isna(r['tempo_dias']) or r['tempo_dias']<=0: flags.append('tempo_ausente_ou_nao_positivo')
        if pd.isna(r['idade_anos']) or not (18<=r['idade_anos']<=110): flags.append('idade_ausente_ou_fora_faixa')
        if pd.isna(r['grupo_estagio']): flags.append('estagio_nao_mapeado_ou_ausente')
        if flags: issues.append({'paciente':r[C['id']], 'motivos':'; '.join(flags)})
    pd.DataFrame(issues,columns=['paciente','motivos']).to_csv(OUT/'exclusoes_e_pendencias.csv',index=False,encoding='utf-8-sig')
    ok=(d['evento'].notna() & d['tempo_dias'].gt(0) & d['idade_anos'].between(18,110)
        & d['grupo_estagio'].notna())
    a=d.loc[ok].copy()
    if len(a)<30 or a['evento'].sum()<15: raise ValueError('Amostra/eventos insuficientes para o ajuste planejado')
    a['idade_10anos']=(a['idade_anos']-65)/10
    a['estagio_II']=(a['grupo_estagio']=='II').astype(int)
    a['estagio_III_IV']=(a['grupo_estagio']=='III-IV').astype(int)
    names=['Idade (por 10 anos)','Estágio II vs I','Estágio III–IV vs I']
    cols=['idade_10anos','estagio_II','estagio_III_IV']
    counts=a.groupby('grupo_estagio',observed=True).agg(pacientes=('evento','size'),obitos=('evento','sum'))
    counts=counts.reindex(['I','II','III-IV']).fillna(0).astype(int)
    counts.to_csv(OUT/'contagens_por_estagio.csv',encoding='utf-8-sig')
    if (counts['pacientes']==0).any(): raise ValueError('Grupo de estágio vazio: revisar categorias')
    model=PHReg(a['tempo_dias'].to_numpy(float),a[cols].to_numpy(float),
                status=a['evento'].to_numpy(int),ties='efron')
    fit=model.fit(disp=0)
    beta=np.asarray(fit.params); se=np.asarray(fit.bse); p=np.asarray(fit.pvalues)
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(se)):
        raise ValueError('Estimativas nao finitas; revisar dados e estabilidade do modelo')
    tab=pd.DataFrame({'variavel':names,'beta':beta,'erro_padrao':se,
                      'HR':np.exp(beta),'IC95_inferior':np.exp(beta-1.96*se),
                      'IC95_superior':np.exp(beta+1.96*se),'p_valor':p})
    tab.to_csv(OUT/'resultados_cox.csv',index=False,encoding='utf-8-sig')
    a[[C['id'],'tempo_dias','evento','idade_anos','grupo_estagio']].to_csv(
        OUT/'coorte_modelagem_cox.csv',index=False,encoding='utf-8-sig')
    # Forest plot com eixo logaritmico, IC95%, linha de nulidade e valores numericos.
    fig,ax=plt.subplots(figsize=(11.6,5.5))
    y=np.arange(len(tab))[::-1]
    colors=[TEAL,ORANGE,RED]
    for i,row in tab.iterrows():
        ax.errorbar(row.HR,y[i],xerr=[[row.HR-row.IC95_inferior],[row.IC95_superior-row.HR]],
                    fmt='o',color=colors[i],ecolor=colors[i],markersize=9,capsize=5,elinewidth=2.3,zorder=4)
    ax.axvline(1,color=MUTED,lw=1.5,ls='--')
    ax.set_xscale('log')
    lo=max(.05,float(tab.IC95_inferior.min())*.65);hi=float(tab.IC95_superior.max())*1.5
    ax.set_xlim(lo,hi);ax.set_ylim(-.6,2.7)
    ax.set_yticks(y,names);ax.tick_params(axis='y',length=0,pad=15)
    ax.set_xlabel('Hazard ratio (escala logarítmica) · IC 95%',labelpad=13)
    ax.grid(axis='x',color=GRID,lw=.9);ax.set_axisbelow(True)
    ax.set_xticks([v for v in (0.25,0.5,1,2,4,8) if lo < v < hi])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    fig.subplots_adjust(left=.26,right=.96,top=.77,bottom=.20)
    fig.text(.05,.94,'Associações com a sobrevida global',fontsize=19,fontweight='bold',color=NAVY)
    fig.text(.05,.875,f'TCGA-PAAD · Cox multivariável · n = {len(a)} · óbitos = {int(a.evento.sum())}',fontsize=11,color=MUTED)
    fig.text(.05,.045,'Estimativas exploratórias; não representam validação preditiva nem causalidade.',fontsize=9,color=MUTED)
    save(fig,'06_forest_plot_cox.png')
    # Painel de elegibilidade e eventos por grupo: denominadores explicitos.
    fig,ax=plt.subplots(figsize=(10.5,5.8))
    labels=['I','II','III–IV'];n=counts.pacientes.to_numpy();e=counts.obitos.to_numpy()
    xx=np.arange(3);ax.bar(xx,n,color=[TEAL,'#3D8195',ORANGE],width=.55,label='Pacientes')
    ax.bar(xx,e,color=NAVY,width=.55,alpha=.72,label='Óbitos registrados')
    for i in range(3):
        ax.text(i,n[i]+max(n)*.025,f'{n[i]} pacientes · {e[i]} óbitos',ha='center',fontsize=10,color=NAVY)
    ax.set_xticks(xx,labels);ax.set_ylabel('Número de pacientes');ax.set_ylim(0,max(n)*1.23)
    ax.grid(axis='y',color=GRID);ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='upper right')
    fig.subplots_adjust(left=.10,right=.96,top=.76,bottom=.20)
    fig.text(.10,.93,'Composição da coorte por estágio',fontsize=19,fontweight='bold',color=NAVY)
    fig.text(.10,.865,'Grupos agregados para análise exploratória; avaliar estabilidade dos grupos pequenos.',fontsize=10,color=MUTED)
    fig.text(.10,.06,'Categorias Stage 0, Stage 0a e estágio ausente não são agrupadas automaticamente.',fontsize=9,color=MUTED)
    save(fig,'06_coorte_por_estagio.png')
    with open(OUT/'relatorio_metodologico.txt','w',encoding='utf-8') as f:
        f.write('TCGA-PAAD | REGRESSAO DE COX EXPLORATORIA\n\n')
        f.write(f'Origem: {SOURCE.name}\nPacientes originais: {len(d)}\n')
        f.write(f'Incluidos: {len(a)}; obitos: {int(a.evento.sum())}; excluidos: {len(d)-len(a)}\n\n')
        f.write('Desfecho: obito (1) ou censura (0); tempos: dias ate obito para falecidos, dias ate ultimo acompanhamento para vivos.\n')
        f.write('Tempo zero assumido conforme campos GDC; validar definicoes temporais na fonte.\n')
        f.write('Idade: por 10 anos; estagio I como referencia; estagio II; estagio III-IV.\n')
        f.write('Estimacao: PHReg (statsmodels), empates Efron, IC Wald 95%.\n')
        f.write('ATENCAO: hipotese de riscos proporcionais NAO testada neste script.\n')
        f.write('Grupos pequenos podem produzir IC amplos/instabilidade; resultados exploratorios.\n')
        f.write('Nao realizar inferencia causal nem afirmar validacao externa ou performance preditiva.\n\n')
        f.write('Contagens:\n'+counts.to_string()+'\n\nResultados:\n'+tab.to_string(index=False)+'\n')
    print('\nMODELO COX - RESULTADOS EXPLORATORIOS')
    print(f'Original: {len(d)} | elegiveis: {len(a)} | eventos: {int(a.evento.sum())} | excluidos: {len(d)-len(a)}')
    print(counts.to_string());print(tab.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nATENCAO: pressuposto de riscos proporcionais ainda nao avaliado.\nSaida:',OUT)

if __name__=='__main__': main()
