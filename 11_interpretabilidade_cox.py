#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etapa 11: interpretabilidade exploratoria Cox, reproduzindo a etapa 10.
Dependencias: numpy pandas scikit-learn lifelines matplotlib.
Coeficiente * covariavel e contribuicao EXATA ao log-risco linear, nao SHAP.
"""
from pathlib import Path
import argparse, json, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from lifelines import CoxPHFitter

BASE=Path(__file__).resolve().parent

def stage_group(value):
    s=str(value).upper().replace('STAGE','').strip()
    if s in ('','NAN','NONE','NOT REPORTED','UNKNOWN','STAGE X','X'): return np.nan
    for x in ('IV','III','II','I'):
        if s.startswith(x): return x
    return np.nan

def clinical(tr,te):
    a=tr[['idade_anos']].copy(); b=te[['idade_anos']].copy()
    med=a['idade_anos'].median()
    if not np.isfinite(med): raise ValueError('Idade ausente no treino')
    a['idade_anos']=a['idade_anos'].fillna(med); b['idade_anos']=b['idade_anos'].fillna(med)
    for c in ('II','III','IV'):
        a['estagio_'+c]=(tr['estagio']==c).astype(float)
        b['estagio_'+c]=(te['estagio']==c).astype(float)
    return a,b

def rna(train,test,ids,n_genes,min_cpm,min_fraction):
    lib_tr=train.sum(axis=0); lib_te=test.sum(axis=0)
    if np.any(lib_tr<=0) or np.any(lib_te<=0): raise ValueError('Biblioteca com profundidade zero')
    cpm_tr=train/lib_tr[None,:]*1e6
    keep=(cpm_tr>=min_cpm).sum(axis=1)>=max(2,int(np.ceil(min_fraction*train.shape[1])))
    if not keep.any(): raise ValueError('Nenhum gene elegivel')
    var=np.var(np.log2(1+cpm_tr[keep]),axis=1)
    chosen=np.where(keep)[0][np.argsort(-var,kind='stable')[:min(n_genes,len(var))]]
    xtr=np.log2(1+train[chosen]/lib_tr[None,:]*1e6).T
    xte=np.log2(1+test[chosen]/lib_te[None,:]*1e6).T
    mu=xtr.mean(axis=0); sd=xtr.std(axis=0,ddof=0); sd=np.where(sd==0,1,sd)
    names=['gene_'+str(i) for i in chosen]
    return pd.DataFrame((xtr-mu)/sd,columns=names),pd.DataFrame((xte-mu)/sd,columns=names),dict(zip(names,[str(ids[i]) for i in chosen]))

def save_plot(data,path,title,xlabel,top=20):
    if data.empty: return
    d=data.head(top).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,max(4,len(d)*.36+1)))
    ax.barh(d['variavel'],d['valor'],color='#1D7775')
    ax.set(xlabel=xlabel,title=title)
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=BASE/'resultados_integracao_09d')
    ap.add_argument('--etapa10',type=Path,default=BASE/'resultados_modelagem_10')
    ap.add_argument('--saida',type=Path,default=BASE/'resultados_interpretabilidade_11')
    args=ap.parse_args(); args.saida.mkdir(parents=True,exist_ok=True)
    cfg_path=args.etapa10/'10_configuracao.json'
    if not cfg_path.is_file(): raise FileNotFoundError(f'Arquivo de configuracao ausente: {cfg_path}')
    cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    splits=int(cfg['dobras']); repeats=int(cfg['repeticoes']); seed=int(cfg['seed'])
    genes=int(cfg['max_genes_por_dobra']); penalty=float(cfg['penalizacao'])
    min_cpm=float(cfg['min_cpm']); fraction=float(cfg['fracao_minima'])
    cp=args.base/'09d_pacientes_pareados_clinicos.csv'; mp=args.base/'09d_contagens_brutas_pareadas.csv.gz'
    for p in (cp,mp):
        if not p.is_file(): raise FileNotFoundError(f'Arquivo ausente: {p}')
    cli=pd.read_csv(cp,low_memory=False)
    needed=['id_paciente','evento_obito','tempo_dias','diagnoses.age_at_diagnosis','diagnoses.ajcc_pathologic_stage']
    if any(c not in cli for c in needed): raise ValueError('Colunas clinicas ausentes: '+str([c for c in needed if c not in cli]))
    cli['id_paciente']=cli.id_paciente.astype(str).str.strip().str.upper()
    if cli.id_paciente.duplicated().any(): raise ValueError('IDs clinicos duplicados')
    cli['evento_obito']=pd.to_numeric(cli.evento_obito,errors='coerce')
    cli['tempo_dias']=pd.to_numeric(cli.tempo_dias,errors='coerce')
    cli['idade_anos']=pd.to_numeric(cli['diagnoses.age_at_diagnosis'],errors='coerce')/365.25
    cli['estagio']=cli['diagnoses.ajcc_pathologic_stage'].map(stage_group)
    cli=cli.loc[cli.evento_obito.isin([0,1]) & cli.tempo_dias.gt(0)]
    cli=cli.loc[cli.idade_anos.between(18,110) & cli.estagio.isin(['I','II','III','IV'])].copy()
    print('ETAPA 11 | Carregando contagens brutas...',flush=True)
    mat=pd.read_csv(mp,index_col=0)
    mat.columns=mat.columns.astype(str).str.strip().str.upper()
    if mat.columns.duplicated().any() or mat.index.duplicated().any(): raise ValueError('IDs duplicados na matriz')
    ids=sorted(set(cli.id_paciente)&set(mat.columns))
    cli=cli.set_index('id_paciente').loc[ids].reset_index(); mat=mat.loc[:,ids]
    arr=mat.to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all() or (arr<0).any() or not np.allclose(arr,np.rint(arr)):
        raise ValueError('Matriz deve conter contagens brutas inteiras e nao negativas')
    if len(ids)!=int(cfg['pacientes_elegiveis']): raise ValueError('Coorte difere da etapa 10; nao e comparavel')
    if int(cli.evento_obito.sum())!=int(cfg['eventos']): raise ValueError('Eventos diferem da etapa 10')
    cv=RepeatedStratifiedKFold(n_splits=splits,n_repeats=repeats,random_state=seed)
    coef_rows=[]; contrib_rows=[]; selection_rows=[]; errors=[]
    for fold,(itr,ite) in enumerate(cv.split(np.zeros(len(cli)),cli.evento_obito.astype(int)),1):
        tr=cli.iloc[itr].reset_index(drop=True); te=cli.iloc[ite].reset_index(drop=True)
        c_tr,c_te=clinical(tr,te)
        try:
            r_tr,r_te,gene_map=rna(arr[:,itr],arr[:,ite],mat.index,genes,min_cpm,fraction)
        except Exception as e:
            errors.append({'dobra':fold,'modelo':'RNA','erro':repr(e)}); continue
        for model_name in ('clinico','transcriptomico','integrado'):
            if model_name=='clinico': xtr,xte=c_tr,c_te
            elif model_name=='transcriptomico': xtr,xte=r_tr,r_te
            else:
                xtr=pd.concat([c_tr,r_tr],axis=1); xte=pd.concat([c_te,r_te],axis=1)
            cols=xtr.columns[xtr.std(axis=0,ddof=0).fillna(0).gt(1e-9)]
            try:
                if not len(cols): raise ValueError('Nenhuma covariavel variavel')
                df=xtr.loc[:,cols].reset_index(drop=True).copy()
                df['tempo']=tr.tempo_dias.to_numpy(float); df['evento']=tr.evento_obito.to_numpy(int)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit=CoxPHFitter(penalizer=penalty)
                    fit.fit(df,duration_col='tempo',event_col='evento',show_progress=False)
                beta=fit.params_.reindex(cols).astype(float)
                # lifelines centraliza as covariaveis: log risco = sum beta*(x - media_treino)
                center=fit._norm_mean.reindex(cols).astype(float)
                vals=xte.loc[:,cols].to_numpy(float)
                components=(vals-center.to_numpy()[None,:])*beta.to_numpy()[None,:]
                predicted=fit.predict_log_partial_hazard(xte.loc[:,cols]).to_numpy().ravel()
                if not np.allclose(components.sum(axis=1),predicted,rtol=1e-6,atol=1e-6):
                    raise ValueError('Contribuicoes nao reproduzem log-risco previsto')
                for col in cols:
                    gid=gene_map.get(col,'')
                    coef_rows.append({'dobra':fold,'modelo':model_name,'variavel':gid or col,
                                      'tipo':'gene' if gid else 'clinica','coeficiente':float(beta[col]),
                                      'hazard_ratio_por_unidade':float(np.exp(np.clip(beta[col],-700,700))),
                                      'media_treino':float(center[col])})
                for i,patient in enumerate(te.id_paciente):
                    for j,col in enumerate(cols):
                        contrib_rows.append({'dobra':fold,'modelo':model_name,'id_paciente':patient,
                                             'variavel':gene_map.get(col,col),'tipo':'gene' if col in gene_map else 'clinica',
                                             'contribuicao_log_risco':float(components[i,j]),
                                             'valor_transformado':float(vals[i,j]),'log_risco_total':float(predicted[i])})
                print(f'Dobra {fold}/{splits*repeats} | {model_name}: {len(cols)} covariaveis',flush=True)
            except Exception as e:
                errors.append({'dobra':fold,'modelo':model_name,'erro':repr(e)})
                print(f'AVISO {fold} {model_name}: {e}',flush=True)
        for gid in gene_map.values(): selection_rows.append({'dobra':fold,'gene_id':gid})
    co=pd.DataFrame(coef_rows,columns=['dobra','modelo','variavel','tipo','coeficiente','hazard_ratio_por_unidade','media_treino'])
    ct=pd.DataFrame(contrib_rows,columns=['dobra','modelo','id_paciente','variavel','tipo','contribuicao_log_risco','valor_transformado','log_risco_total'])
    sel=pd.DataFrame(selection_rows,columns=['dobra','gene_id'])
    co.to_csv(args.saida/'11_coeficientes_por_dobra.csv',index=False,encoding='utf-8-sig')
    ct.to_csv(args.saida/'11_contribuicoes_individuais_teste.csv.gz',index=False,compression='gzip')
    sel.to_csv(args.saida/'11_genes_selecionados.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(errors,columns=['dobra','modelo','erro']).to_csv(args.saida/'11_falhas.csv',index=False,encoding='utf-8-sig')
    if not co.empty:
        summary=(co.groupby(['modelo','variavel','tipo'],as_index=False)
                 .agg(n_dobras=('dobra','nunique'),coeficiente_medio=('coeficiente','mean'),
                      coeficiente_dp=('coeficiente','std'),fracao_coef_positivo=('coeficiente',lambda s:float((s>0).mean()))))
        summary.to_csv(args.saida/'11_estabilidade_coeficientes.csv',index=False,encoding='utf-8-sig')
    else: summary=pd.DataFrame()
    if not ct.empty:
        imp=(ct.assign(abs_contrib=lambda d:d.contribuicao_log_risco.abs())
             .groupby(['modelo','variavel','tipo'],as_index=False)
             .agg(n_observacoes_teste=('abs_contrib','size'),contribuicao_absoluta_media=('abs_contrib','mean')))
        imp=imp.sort_values(['modelo','contribuicao_absoluta_media'],ascending=[True,False])
        imp.to_csv(args.saida/'11_importancia_descritiva.csv',index=False,encoding='utf-8-sig')
        for name in ('clinico','transcriptomico','integrado'):
            d=imp[imp.modelo==name].rename(columns={'contribuicao_absoluta_media':'valor'})
            save_plot(d,args.saida/f'11_importancia_{name}.png',f'Contribuicoes absolutas medias - {name}',
                      'Media |contribuicao ao log-risco| nos testes')
    else: imp=pd.DataFrame()
    if not sel.empty:
        freq=sel.groupby('gene_id',as_index=False).agg(n_dobras=('dobra','nunique')).sort_values('n_dobras',ascending=False)
        freq.to_csv(args.saida/'11_frequencia_selecao_genes.csv',index=False,encoding='utf-8-sig')
        save_plot(freq.rename(columns={'gene_id':'variavel','n_dobras':'valor'}),args.saida/'11_frequencia_genes.png',
                  'Frequencia de selecao por variancia','Numero de dobras')
    else: freq=pd.DataFrame()
    overview={'etapa':'11','status':'concluida' if not errors else 'incompleta',
              'pacientes_elegiveis':len(ids),'dobras_previstas':splits*repeats,
              'modelos':3,'ajustes_concluidos':len(co.groupby(['dobra','modelo'])) if not co.empty else 0,
              'falhas':len(errors),'genes_distintos':len(freq),
              'metodo':'coeficiente Cox x (covariavel - media treino); decomposicao exata do log-risco',
              'nota':'nao e SHAP; nao representa efeito causal ou biomarcador validado'}
    (args.saida/'11_resumo.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False),encoding='utf-8')
    report=('ETAPA 11 | INTERPRETABILIDADE EXPLORATORIA COX | TCGA-PAAD\n\n'+
            json.dumps(overview,indent=2,ensure_ascii=False)+'\n\n'+
            'MODELOS: os mesmos splits, selecao de genes por variancia no treino e penalizacao da etapa 10.\n'+
            'Contribuicoes: beta_j * (x_j - media_treino_j). A soma reproduz o log-risco previsto pelo lifelines.\n'+
            'Valores de genes sao log2(CPM+1) padronizados por parametros do treino; idade em anos, estagios dummies.\n'+
            'Hazard ratios nao sao comparaveis diretamente entre genes padronizados e idade em anos.\n'+
            'Contribuicoes absolutas medias sao descritivas e dependem da distribuicao das covariaveis.\n'+
            'A selecao por variancia nao prova associacao prognostica nem relevancia biologica.\n'+
            'As dobras repetidas sao dependentes; frequencias e desvios nao constituem IC independentes.\n'+
            'Identificadores Ensembl mantidos sem anotacao externa nao verificada.\n'+
            'Nao houve analise de vias, SHAP, validacao externa, inferencia causal ou validacao clinica.\n'+
            'Revisar lote, pureza tumoral, qualidade, censura e origem temporal antes de interpretacao biologica.\n')
    (args.saida/'11_relatorio_metodologico.txt').write_text(report,encoding='utf-8')
    print('\n'+report+'\nResultados:',args.saida)
    if errors or overview['ajustes_concluidos']!=splits*repeats*3: sys.exit(2)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERRO ETAPA 11: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
