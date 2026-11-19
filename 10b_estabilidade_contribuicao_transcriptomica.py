#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETAPA 10B — auditoria de estabilidade e contribuição incremental do RNA-seq.

Analisa resultados da etapa 10 SEM considerar dobras repetidas independentes.
Opcionalmente executa cenários de sensibilidade com o script 10 original.
Dependências: pandas numpy matplotlib; cenários opcionais exigem dependências da etapa 10.
"""
from pathlib import Path
import argparse, json, subprocess, sys
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
MODELOS=['clinico','transcriptomico','integrado']
PAIRS=[('integrado','clinico'),('transcriptomico','clinico'),('integrado','transcriptomico')]

def salvar(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')

def analisar(pasta,saida,prefixo='10b'):
    path=pasta/'10_resultados_por_dobra.csv'
    if not path.is_file(): raise FileNotFoundError(f'Arquivo obrigatório ausente: {path}')
    df=pd.read_csv(path)
    req={'dobra','modelo','c_index'}
    if not req.issubset(df.columns): raise ValueError(f'Colunas ausentes em {path}: {sorted(req-set(df.columns))}')
    df['c_index']=pd.to_numeric(df.c_index,errors='coerce')
    if df[['dobra','modelo']].duplicated().any(): raise ValueError('Duplicações de modelo por dobra.')
    if df.c_index.isna().any() or not df.c_index.between(0,1).all(): raise ValueError('C-index inválido/ausente.')
    if not set(MODELOS).issubset(set(df.modelo)): raise ValueError('Nem todos os três modelos estão presentes.')
    wide=df.pivot(index='dobra',columns='modelo',values='c_index').sort_index()
    if wide[MODELOS].isna().any().any(): raise ValueError('Dobras incompletas: comparação pareada impossível.')
    if len(wide)<2: raise ValueError('São necessárias ao menos duas dobras completas.')
    cfg_path=pasta/'10_configuracao.json'
    cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig')) if cfg_path.exists() else {}
    expected=cfg.get('dobras',0)*cfg.get('repeticoes',0)
    if expected and len(wide)!=expected: raise ValueError(f'Esperadas {expected} dobras; encontradas {len(wide)}.')
    falhas=pasta/'10_falhas.csv'
    if falhas.exists() and len(pd.read_csv(falhas))>0: raise ValueError('Há falhas registradas na etapa 10.')
    resumo=(df.groupby('modelo').c_index.agg(n_dobras='count',media='mean',dp='std',mediana='median',minimo='min',maximo='max').reset_index())
    salvar(resumo,saida/f'{prefixo}_resumo_estabilidade.csv')
    diff_rows=[]; diffs=pd.DataFrame({'dobra':wide.index})
    for a,b in PAIRS:
        d=(wide[a]-wide[b]).to_numpy()
        nome=f'{a}_menos_{b}'
        diffs[nome]=d
        diff_rows.append({'comparacao':f'{a} - {b}','n_dobras_dependentes':len(d),
                          'delta_medio_descritivo':float(d.mean()),'delta_mediano':float(np.median(d)),
                          'delta_dp_descritivo':float(d.std(ddof=1)),
                          'dobras_delta_positivo':int((d>0).sum()),'dobras_delta_negativo':int((d<0).sum()),
                          'dobras_delta_zero':int((d==0).sum())})
    salvar(diffs,saida/f'{prefixo}_diferencas_pareadas_por_dobra.csv')
    comp=pd.DataFrame(diff_rows)
    salvar(comp,saida/f'{prefixo}_contribuicao_incremental.csv')
    genes_path=pasta/'10_genes_selecionados_por_dobra.csv'
    gene_info={'disponivel':False}
    if genes_path.exists():
        genes=pd.read_csv(genes_path,dtype={'gene_id':str})
        if {'dobra','gene_id'}.issubset(genes.columns) and len(genes):
            genes=genes.dropna(subset=['dobra','gene_id']).drop_duplicates(['dobra','gene_id'])
            sets={k:set(g.gene_id.astype(str)) for k,g in genes.groupby('dobra')}
            if set(sets)!=set(wide.index): raise ValueError('Dobras de seleção gênica não coincidem com as da modelagem.')
            freq=Counter(g for ss in sets.values() for g in ss)
            ranking=pd.DataFrame([{'gene_id':g,'n_dobras':n,'frequencia_dobras':n/len(sets)} for g,n in freq.items()])
            ranking=ranking.sort_values(['n_dobras','gene_id'],ascending=[False,True])
            salvar(ranking,saida/f'{prefixo}_frequencia_genes.csv')
            jac=[]; keys=sorted(sets)
            for i,a in enumerate(keys):
                for b in keys[i+1:]:
                    inter=len(sets[a]&sets[b]); union=len(sets[a]|sets[b])
                    jac.append({'dobra_a':a,'dobra_b':b,'intersecao':inter,'uniao':union,'jaccard':inter/union if union else np.nan})
            salvar(pd.DataFrame(jac),saida/f'{prefixo}_sobreposicao_genes.csv')
            gene_info={'disponivel':True,'genes_distintos':len(freq),'genes_presentes_em_todas_dobras':sum(n==len(sets) for n in freq.values()),
                       'jaccard_medio_descritivo':float(np.mean([r['jaccard'] for r in jac])) if jac else None,
                       'max_genes_por_dobra_observado':max(map(len,sets.values()))}
    fig,ax=plt.subplots(figsize=(9,5))
    for m in MODELOS: ax.plot(wide.index,wide[m],marker='o',label=m)
    ax.axhline(.5,color='gray',ls='--',lw=1,label='Referência 0,5')
    ax.set(xlabel='Dobra (repetições dependentes)',ylabel='C-index no teste',title='TCGA-PAAD | Estabilidade entre dobras')
    ax.legend(); ax.grid(alpha=.2);fig.tight_layout();fig.savefig(saida/f'{prefixo}_estabilidade_dobras.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for col in diffs.columns[1:]: ax.plot(diffs.dobra,diffs[col],marker='o',label=col.replace('_menos_',' − '))
    ax.axhline(0,color='gray',ls='--');ax.set(xlabel='Dobra',ylabel='Diferença pareada de C-index',title='TCGA-PAAD | Diferenças exploratórias')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(saida/f'{prefixo}_deltas_pareados.png',dpi=220);plt.close(fig)
    resultado={'etapa':'10B','status':'análise descritiva concluída','n_dobras_dependentes':len(wide),
               'configuracao_etapa10':cfg,'resumo_modelos':resumo.to_dict('records'),
               'comparacoes':comp.to_dict('records'),'estabilidade_genes':gene_info,
               'alerta':'Dobras repetidas compartilham pacientes: DP não é IC; não realizar testes ingênuos nem declarar superioridade estatística.'}
    (saida/f'{prefixo}_resumo.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf-8')
    linhas=['ETAPA 10B | TCGA-PAAD | ESTABILIDADE E CONTRIBUIÇÃO INCREMENTAL','',
            'ANÁLISE DESCRITIVA DE DOBRAS DEPENDENTES','',resumo.to_string(index=False),'',
            'DIFERENÇAS PAREADAS (MESMAS DOBRAS)','',comp.to_string(index=False),'',
            'ESTABILIDADE DE SELEÇÃO GÊNICA','',json.dumps(gene_info,ensure_ascii=False,indent=2),'',
            'LIMITAÇÕES','As mesmas pessoas podem aparecer em testes de repetições diferentes.',
            'As dobras não são observações independentes: DP entre dobras não é IC 95%.',
            'Sem teste de significância, superioridade estatística ou validação externa.',
            'Seleção por variância não implica relevância biológica ou causal.',
            'Comparações de sensibilidade são exploratórias e não constituem otimização de hiperparâmetros.',
            'Revisar temporalidade, censura, origem dos dados, lote e qualidade biológica antes de conclusões.']
    (saida/f'{prefixo}_relatorio_metodologico.txt').write_text('\n'.join(linhas),encoding='utf-8')
    print('\n'.join(linhas));return resultado

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entrada',type=Path,default=ROOT/'resultados_modelagem_10')
    p.add_argument('--saida',type=Path,default=ROOT/'resultados_modelagem_10b')
    p.add_argument('--sensibilidade',action='store_true',help='Reexecutar etapa 10 com penalizações pré-especificadas 0.5 e 2.0; pode demorar.')
    p.add_argument('--script10',type=Path,default=ROOT/'10_modelagem_clinica_transcriptomica.py')
    args=p.parse_args();args.saida.mkdir(parents=True,exist_ok=True)
    resultado=analisar(args.entrada,args.saida)
    if args.sensibilidade:
        cfg=resultado['configuracao_etapa10']
        if not args.script10.exists(): raise FileNotFoundError(f'Script 10 ausente: {args.script10}')
        required=['max_genes_por_dobra','dobras','repeticoes','min_cpm','fracao_minima']
        if any(k not in cfg for k in required): raise ValueError('Configuração da etapa 10 incompleta para sensibilidade.')
        sens=[]
        for penal in [0.5,2.0]:
            dest=args.saida/f'sensibilidade_penalizacao_{str(penal).replace(".","_")}'
            cmd=[sys.executable,str(args.script10),'--saida',str(dest),'--genes',str(cfg['max_genes_por_dobra']),
                 '--dobras',str(cfg['dobras']),'--repeticoes',str(cfg['repeticoes']),
                 '--penalizacao',str(penal),'--min-cpm',str(cfg['min_cpm']),
                 '--fracao-minima',str(cfg['fracao_minima'])]
            print(f'\nExecutando sensibilidade penalização={penal}',flush=True)
            proc=subprocess.run(cmd,check=False)
            if proc.returncode!=0: raise RuntimeError(f'Etapa 10 falhou no cenário {penal}; código {proc.returncode}')
            sub=analisar(dest,dest,prefixo='10b')
            for item in sub['resumo_modelos']:
                sens.append({'penalizacao':penal,'modelo':item['modelo'],'c_index_medio':item['media'],'n_dobras':item['n_dobras']})
        salvar(pd.DataFrame(sens),args.saida/'10b_sensibilidade_penalizacao.csv')
        print('Sensibilidade concluída; resultados apenas exploratórios.')
    print('\nArquivos salvos em:',args.saida)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERRO ETAPA 10B: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
