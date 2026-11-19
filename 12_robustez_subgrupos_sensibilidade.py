#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etapa 12 TCGA-PAAD: auditoria de robustez DESCRITIVA das previsoes OOF da etapa 10.
Nao reajusta modelos, nao faz inferencia independente entre dobras repetidas.
Requer: pandas, numpy, lifelines, matplotlib.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

ROOT=Path(__file__).resolve().parent
MODELS=['clinico','transcriptomico','integrado']

def stage_group(value):
    s=str(value).upper().replace('STAGE','').strip()
    if s in ('','NAN','NONE','NOT REPORTED','UNKNOWN','X'): return None
    for x in ('IV','III','II','I'):
        if s.startswith(x): return x
    return None

def ci(df):
    if len(df)<5 or df.evento_obito.sum()<2 or (df.evento_obito==0).sum()<1: return np.nan
    try:
        return float(concordance_index(df.tempo_dias.to_numpy(float),-df.risco_log_relativo.to_numpy(float),df.evento_obito.to_numpy(int)))
    except (ValueError,ZeroDivisionError): return np.nan

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--etapa10',type=Path,default=ROOT/'resultados_modelagem_10')
    ap.add_argument('--base',type=Path,default=ROOT/'resultados_integracao_09d')
    ap.add_argument('--saida',type=Path,default=ROOT/'resultados_robustez_12')
    ap.add_argument('--idade-corte',type=float,default=65)
    ap.add_argument('--min-subgrupo',type=int,default=10)
    a=ap.parse_args()
    p=a.etapa10/'10_predicoes_fora_amostra.csv'; q=a.base/'09d_pacientes_pareados_clinicos.csv'
    for f in (p,q):
        if not f.is_file(): raise FileNotFoundError(f'Arquivo obrigatorio ausente: {f}')
    pred=pd.read_csv(p,encoding='utf-8-sig'); clin=pd.read_csv(q,encoding='utf-8-sig',low_memory=False)
    required_pred=['dobra','modelo','id_paciente','tempo_dias','evento_obito','risco_log_relativo']
    required_cli=['id_paciente','diagnoses.age_at_diagnosis','diagnoses.ajcc_pathologic_stage']
    for name,df,cols in [('predicoes',pred,required_pred),('clinica',clin,required_cli)]:
        missing=set(cols)-set(df.columns)
        if missing: raise ValueError(f'{name}: colunas ausentes: {sorted(missing)}')
    for df in (pred,clin): df['id_paciente']=df.id_paciente.astype(str).str.strip().str.upper()
    if clin.id_paciente.duplicated().any(): raise ValueError('IDs clinicos duplicados')
    if pred.duplicated(['dobra','modelo','id_paciente']).any(): raise ValueError('Previsoes duplicadas na mesma dobra/modelo')
    if not set(pred.modelo.unique()).issubset(set(MODELS)): raise ValueError('Modelos desconhecidos')
    if pred[['tempo_dias','evento_obito','risco_log_relativo']].isna().any().any(): raise ValueError('Previsoes com dados ausentes')
    for col in ('tempo_dias','evento_obito','risco_log_relativo'):
        pred[col]=pd.to_numeric(pred[col],errors='raise')
    if not np.isfinite(pred[['tempo_dias','evento_obito','risco_log_relativo']].to_numpy(float)).all(): raise ValueError('Previsoes nao finitas')
    if not pred.evento_obito.isin([0,1]).all() or not pred.tempo_dias.gt(0).all(): raise ValueError('Tempo/evento invalidos')
    clin['idade_anos']=pd.to_numeric(clin['diagnoses.age_at_diagnosis'],errors='coerce')/365.25
    clin['estagio']=clin['diagnoses.ajcc_pathologic_stage'].map(stage_group)
    df=pred.merge(clin[['id_paciente','idade_anos','estagio']],on='id_paciente',how='left',validate='many_to_one',indicator=True)
    if (df['_merge']!='both').any(): raise ValueError('Previsoes sem vinculo clinico')
    df=df.drop(columns='_merge')
    if df.idade_anos.isna().any() or df.estagio.isna().any(): raise ValueError('Subgrupos ausentes; revisar coorte')
    df['faixa_idade']=np.where(df.idade_anos<a.idade_corte,f'<{a.idade_corte:g} anos',f'>={a.idade_corte:g} anos')
    df['grupo_estagio']=np.where(df.estagio.isin(['I','II']),'I-II','III-IV')
    # A consistencia entre os modelos e essencial para diferencas pareadas.
    key=['dobra','id_paciente']
    pivot=df.pivot(index=key,columns='modelo',values='risco_log_relativo')
    if pivot[MODELS].isna().any().any(): raise ValueError('Modelos sem as mesmas observacoes de teste')
    out=a.saida; out.mkdir(parents=True,exist_ok=True)
    # C-index por dobra, modelo e estrato. Nao agregar pacientes de dobras repetidas como independentes.
    rows=[]
    for (fold,model),g in df.groupby(['dobra','modelo'],sort=True):
        for tipo,col in [('geral',None),('idade','faixa_idade'),('estagio','grupo_estagio'),('evento','evento_obito')]:
            for label,h in ([('todos',g)] if col is None else list(g.groupby(col))):
                n=len(h); score=ci(h) if n>=a.min_subgrupo else np.nan
                rows.append(dict(dobra=int(fold),modelo=model,estrato=tipo,grupo=str(label),n=n,eventos=int(h.evento_obito.sum()),censuras=int((h.evento_obito==0).sum()),c_index=score,avaliavel=bool(np.isfinite(score))))
    sub=pd.DataFrame(rows)
    sub.to_csv(out/'12_cindex_subgrupos_por_dobra.csv',index=False,encoding='utf-8-sig')
    summary=sub.groupby(['modelo','estrato','grupo'],as_index=False).agg(n_dobras=('c_index','count'),c_index_medio=('c_index','mean'),c_index_dp_descritivo=('c_index','std'),n_mediano=('n','median'),eventos_medianos=('eventos','median'))
    summary.to_csv(out/'12_resumo_subgrupos.csv',index=False,encoding='utf-8-sig')
    # Diferencas somente dentro da mesma dobra; repeticoes dependentes.
    general=sub[sub.estrato=='geral'].pivot(index='dobra',columns='modelo',values='c_index')
    pairs=[('integrado','clinico'),('transcriptomico','clinico'),('integrado','transcriptomico')]
    changes=[]
    for m1,m0 in pairs:
        if m1 not in general or m0 not in general: continue
        for fold,delta in (general[m1]-general[m0]).items():
            changes.append(dict(dobra=int(fold),comparacao=f'{m1} - {m0}',delta_c_index=float(delta)))
    delta=pd.DataFrame(changes)
    delta.to_csv(out/'12_diferencas_pareadas.csv',index=False,encoding='utf-8-sig')
    delta_summary=delta.groupby('comparacao',as_index=False).agg(n_dobras=('delta_c_index','count'),delta_medio=('delta_c_index','mean'),delta_mediano=('delta_c_index','median'),delta_dp_descritivo=('delta_c_index','std'),dobras_positivas=('delta_c_index',lambda x:int((x>0).sum())),dobras_negativas=('delta_c_index',lambda x:int((x<0).sum())))
    delta_summary.to_csv(out/'12_resumo_diferencas.csv',index=False,encoding='utf-8-sig')
    # Sensibilidade a retirar UMA observacao do teste; SEM refazer ajuste.
    loo=[]
    for (fold,model),g in df.groupby(['dobra','modelo']):
        baseline=ci(g)
        if not np.isfinite(baseline): continue
        for idx,r in g.iterrows():
            s=ci(g.drop(index=idx))
            if np.isfinite(s): loo.append(dict(dobra=int(fold),modelo=model,id_paciente=r.id_paciente,c_index_original=baseline,c_index_sem_paciente=s,delta_remocao=s-baseline,evento=int(r.evento_obito)))
    influence=pd.DataFrame(loo)
    influence.to_csv(out/'12_influencia_remocao_teste.csv',index=False,encoding='utf-8-sig')
    # Censura: tabela de distribuicao, sem criar endpoints alternativos artificiais.
    censor=df[df.modelo=='clinico'].groupby('dobra',as_index=False).agg(n=('id_paciente','size'),eventos=('evento_obito','sum'),tempo_mediano=('tempo_dias','median'))
    censor['censuras']=censor.n-censor.eventos
    censor['fracao_censura']=censor.censuras/censor.n
    censor.to_csv(out/'12_auditoria_censura.csv',index=False,encoding='utf-8-sig')
    # Figuras: apenas estatisticas descritivas.
    fig,ax=plt.subplots(figsize=(9,4.5))
    for model in MODELS:
        d=general[model].dropna() if model in general else pd.Series(dtype=float)
        if len(d): ax.plot(d.index,d.values,marker='o',label=model)
    ax.axhline(.5,color='gray',linestyle='--',alpha=.65)
    ax.set(xlabel='Dobra de teste',ylabel='C-index',title='Etapa 12 | Discriminacao por dobra')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'12_cindex_dobras.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for name,g in delta.groupby('comparacao'):
        ax.plot(g.dobra,g.delta_c_index,marker='o',label=name)
    ax.axhline(0,color='gray',linestyle='--'); ax.set(xlabel='Dobra',ylabel='Diferenca de C-index',title='Comparacoes pareadas descritivas')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'12_diferencas_pareadas.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ss=summary[(summary.estrato.isin(['idade','estagio']))&(summary.n_dobras>0)].copy()
    if len(ss):
        ss['rotulo']=ss.estrato+': '+ss.grupo+' | '+ss.modelo
        ax.barh(ss.rotulo,ss.c_index_medio,color='#217c79'); ax.axvline(.5,color='gray',linestyle='--'); ax.set_xlim(0,1)
    ax.set(xlabel='C-index medio entre dobras avaliaveis',title='Subgrupos | Exploratorio')
    fig.set_size_inches(10,max(5,len(ss)*.33)); fig.tight_layout(); fig.savefig(out/'12_subgrupos.png',dpi=200); plt.close(fig)
    top=influence.assign(magnitude=lambda x:x.delta_remocao.abs()).sort_values('magnitude',ascending=False).head(20)
    top.to_csv(out/'12_maiores_influencias.csv',index=False,encoding='utf-8-sig')
    report={
      'etapa':'12','status':'concluida (auditoria descritiva)','observacoes_predicao':len(df),
      'pacientes_distintos':int(df.id_paciente.nunique()),'dobras':int(df.dobra.nunique()),
      'modelos':sorted(df.modelo.unique().tolist()),'dobras_subgrupos_avaliaveis':int(sub.avaliavel.sum()),
      'dobras_subgrupos_nao_avaliaveis':int((~sub.avaliavel).sum()),
      'comparacoes':delta_summary.to_dict(orient='records'),
      'avisos':['Predicoes da etapa 10 reutilizadas; nenhum modelo foi reajustado.',
                'Dobras repetidas compartilham pacientes; DP nao e IC nem teste de significancia.',
                'Subgrupos pequenos podem ser nao avaliaveis; estadio III-IV e potencialmente escasso.',
                'Analise de influencia remove observacoes APENAS do teste, nao e influencia no ajuste.',
                'Nao ha avaliacao de lote, pureza tumoral, temporalidade ou validacao externa.',
                'C-index para evento isolado nao e calculavel; estrato por evento pode ser NA.',
                'Resultados nao demonstram beneficio clinico, causalidade ou superioridade estatistica.']}
    (out/'12_resumo.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    txt=('ETAPA 12 | ROBUSTEZ DESCRITIVA | TCGA-PAAD\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n\nDIFERENCAS PAREADAS\n'+delta_summary.to_string(index=False)+'\n\nSUBGRUPOS\n'+summary.to_string(index=False))
    (out/'12_relatorio_metodologico.txt').write_text(txt,encoding='utf-8')
    print(txt+'\n\nArquivos em: '+str(out))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERRO ETAPA 12: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
