#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etapa 09B TCGA-PAAD: download opcional, integridade e matriz STAR counts.
Python >=3.9. pip install pandas numpy matplotlib requests
Uso: python 09b_processamento_rnaseq.py --baixar
Ou: python 09b_processamento_rnaseq.py --diretorio-arquivos arquivos_rnaseq
A matriz gerada contém contagens brutas (unstranded), sem normalização ou treino.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ORIGEM = BASE / 'resultados_integracao_09'
SAIDA = BASE / 'resultados_rnaseq_09b'
AZUL, VERDE, LARANJA = '#203A53', '#17877E', '#CB8745'

def salvar(df, nome):
    df.to_csv(SAIDA / nome, index=False, encoding='utf-8-sig')

def md5sum(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for bloco in iter(lambda:f.read(1024*1024), b''): h.update(bloco)
    return h.hexdigest()

def baixar(file_id, destino, sessao, tentativas=3):
    url=f'https://api.gdc.cancer.gov/data/{file_id}'
    parcial=destino.with_name(destino.name+'.part')
    for tentativa in range(tentativas):
        try:
            with sessao.get(url,stream=True,timeout=(20,180)) as r:
                r.raise_for_status()
                ct=r.headers.get('Content-Type','').lower()
                if 'json' in ct: raise RuntimeError('API retornou JSON em vez de TSV; verificar acesso ao arquivo')
                with parcial.open('wb') as f:
                    for bloco in r.iter_content(1024*1024):
                        if bloco: f.write(bloco)
            parcial.replace(destino)
            return
        except Exception:
            parcial.unlink(missing_ok=True)
            if tentativa==tentativas-1: raise
            time.sleep(2*(tentativa+1))

def ler_star(path):
    # GDC STAR gene counts: gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, ...
    df=pd.read_csv(path,sep='\t',comment='#',dtype={'gene_id':str},low_memory=False)
    df.columns=df.columns.astype(str).str.strip()
    if not {'gene_id','unstranded'}.issubset(df.columns):
        raise ValueError(f'Formato STAR counts inesperado: {list(df.columns)[:12]}')
    x=df[['gene_id','unstranded']].copy()
    x['gene_id']=x['gene_id'].astype(str).str.strip()
    x=x[x['gene_id'].str.match(r'^ENSG\d+(?:\.\d+)?$',na=False)]
    if x.empty: raise ValueError('Nenhum identificador gênico ENSG válido')
    if x['gene_id'].duplicated().any(): raise ValueError('Identificadores gênicos duplicados no arquivo')
    numeros=pd.to_numeric(x['unstranded'],errors='coerce')
    if numeros.isna().any() or (numeros<0).any() or not np.allclose(numeros, np.rint(numeros)):
        raise ValueError('Contagens unstranded ausentes, negativas ou não inteiras')
    return pd.Series(numeros.to_numpy(dtype=np.int64),index=x['gene_id'],name=path.name)

def grafico_cobertura(total,confirmados,validos):
    fig,ax=plt.subplots(figsize=(10,5.3))
    nomes=['Coorte clínica','Com RNA primário (metadados)','RNA-seq processado e válido']
    vals=[total,confirmados,validos]
    barras=ax.barh(nomes[::-1],vals[::-1],color=[VERDE,LARANJA,AZUL],height=.56)
    for b in barras: ax.text(b.get_width()+2,b.get_y()+b.get_height()/2,str(round(b.get_width())),va='center',color=AZUL,fontsize=11)
    ax.set_xlim(0,max(vals)*1.2 if max(vals) else 1)
    ax.set_xlabel('Pacientes');ax.set_title('TCGA-PAAD | Cobertura da integração RNA-seq',fontweight='bold',color=AZUL,pad=18)
    ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
    for s in ['top','right','left']:ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.35,bottom=.15,top=.85,right=.93)
    fig.savefig(SAIDA/'09b_cobertura.png',dpi=300,facecolor='white');plt.close(fig)

def grafico_bibliotecas(qc):
    if qc.empty:return
    fig,ax=plt.subplots(figsize=(10,5.4))
    ax.hist(np.log10(qc['total_contagens'].to_numpy()+1),bins=min(22,max(5,int(np.sqrt(len(qc))))),color=VERDE,edgecolor='white')
    ax.set_xlabel('log10(total de contagens unstranded + 1)')
    ax.set_ylabel('Número de amostras')
    ax.set_title('TCGA-PAAD | Profundidade de biblioteca — auditoria',color=AZUL,fontweight='bold',pad=18)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(SAIDA/'09b_bibliotecas.png',dpi=300,facecolor='white');plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baixar',action='store_true',help='Baixa os arquivos do GDC (pode consumir centenas de MB)')
    p.add_argument('--diretorio-arquivos',type=Path,default=BASE/'arquivos_rnaseq')
    p.add_argument('--limite',type=int,default=0,help='Limitar arquivos para teste; 0 = todos')
    p.add_argument('--sem-md5',action='store_true',help='Desativa conferência MD5 (não recomendado)')
    args=p.parse_args()
    SAIDA.mkdir(exist_ok=True)
    args.diretorio_arquivos.mkdir(parents=True,exist_ok=True)
    mapa_path=ORIGEM/'09_pares_clinico_rnaseq.csv'
    manifesto_path=ORIGEM/'09_inventario_manifesto.csv'
    if not mapa_path.exists() or not manifesto_path.exists():
        raise FileNotFoundError('Execute a etapa 09 com --consultar-gdc na mesma pasta antes da etapa 09B')
    pares=pd.read_csv(mapa_path,dtype=str).fillna('')
    man=pd.read_csv(manifesto_path,dtype=str).fillna('')
    requer_p={'id','id_paciente','id_amostra','tipo_amostra'}
    requer_m={'id','filename','md5','size'}
    if not requer_p.issubset(pares.columns) or not requer_m.issubset(man.columns):
        raise ValueError('Colunas inesperadas nos resultados da etapa 09')
    pares=pares[pares['tipo_amostra'].str.casefold().eq('primary tumor')].copy()
    if pares.empty:raise ValueError('Não existem amostras Primary Tumor confirmadas')
    pares=pares.merge(man[['id','filename','md5','size']],on='id',how='left',validate='many_to_one')
    if pares['filename'].isna().any():raise ValueError('IDs do pareamento sem correspondência no manifesto')
    # Conservador: nenhuma escolha arbitrária entre amostras/arquivos repetidos.
    conflitos=pares[pares.duplicated('id_paciente',keep=False)|pares.duplicated('id',keep=False)]
    salvar(conflitos,'09b_conflitos_amostras.csv')
    pares=pares[~pares['id_paciente'].isin(conflitos['id_paciente'])].drop_duplicates('id').copy()
    pares=pares.sort_values('id_paciente')
    if args.limite>0:pares=pares.head(args.limite)
    print(f'Arquivos selecionados: {len(pares)}; pacientes conflitantes excluídos: {conflitos.id_paciente.nunique()}')
    if args.baixar:
        import requests
        sessao=requests.Session()
    else:sessao=None
    registros=[];series=[];ids_referencia=None;sucessos=[]
    for i,r in enumerate(pares.itertuples(index=False),1):
        fid=r.id; nome=r.filename
        pasta=args.diretorio_arquivos/fid
        local=pasta/nome
        # Também aceita arquivo plano no diretório informado.
        if not local.exists() and (args.diretorio_arquivos/nome).exists():local=args.diretorio_arquivos/nome
        linha={'id_arquivo':fid,'id_paciente':r.id_paciente,'id_amostra':r.id_amostra,'arquivo':nome,'status':'','detalhe':''}
        try:
            if not local.exists():
                if not args.baixar:
                    linha['status']='não_baixado';registros.append(linha);continue
                pasta.mkdir(parents=True,exist_ok=True)
                local=pasta/nome
                baixar(fid,local,sessao)
            esperado=int(r.size)
            if local.stat().st_size!=esperado:raise ValueError(f'Tamanho divergente: {local.stat().st_size} vs {esperado}')
            if not args.sem_md5 and md5sum(local).lower()!=r.md5.lower():
                raise ValueError('MD5 divergente do manifesto; arquivo não utilizado')
            counts=ler_star(local)
            if ids_referencia is None:ids_referencia=counts.index
            elif not counts.index.equals(ids_referencia):
                raise ValueError('Ordem ou conjunto de gene IDs diferente da primeira amostra; não integrar automaticamente')
            counts.name=r.id_paciente
            series.append(counts)
            linha.update(status='válido',n_genes=len(counts),total_contagens=int(counts.sum()),genes_nao_zero=int((counts>0).sum()))
            sucessos.append({'id_paciente':r.id_paciente,'id_amostra':r.id_amostra,'id_arquivo':fid})
        except Exception as e:
            linha['status']='erro';linha['detalhe']=f'{type(e).__name__}: {e}'
        registros.append(linha)
        if i%10==0 or i==len(pares):print(f'Processados {i}/{len(pares)} | válidos: {len(series)}')
    audit=pd.DataFrame(registros)
    salvar(audit,'09b_auditoria_arquivos.csv')
    validos=audit[audit.status.eq('válido')].copy()
    salvar(validos,'09b_controle_qualidade.csv')
    salvar(pd.DataFrame(sucessos),'09b_pacientes_amostras_validos.csv')
    if series:
        matriz=pd.concat(series,axis=1)
        matriz.index.name='gene_id'
        matriz.to_csv(SAIDA/'09b_matriz_contagens_brutas.csv.gz',compression='gzip',encoding='utf-8')
        genes=pd.DataFrame({'gene_id':matriz.index,'n_pacientes_expressando':(matriz>0).sum(axis=1).to_numpy(),
                            'soma_contagens':matriz.sum(axis=1).to_numpy()})
        salvar(genes,'09b_resumo_genes.csv')
        grafico_bibliotecas(validos)
    total_clinico=185
    resumo09=ORIGEM/'09_resumo.json'
    if resumo09.exists():
        d=json.loads(resumo09.read_text(encoding='utf-8'))
        total_clinico=int(d.get('pacientes_clinicos',185))
        n_meta=int(d.get('pacientes_com_rna_tumor_primario',len(pares)))
    else:n_meta=len(pares)
    grafico_cobertura(total_clinico,n_meta,len(series))
    resumo={'etapa':'09B','arquivos_selecionados':len(pares),'arquivos_validos':len(series),
            'arquivos_nao_baixados':int((audit.status=='não_baixado').sum()),
            'arquivos_com_erro':int((audit.status=='erro').sum()),
            'pacientes_com_matriz':len(series),'genes_na_matriz':len(ids_referencia) if ids_referencia is not None else 0,
            'contagem':'STAR unstranded bruta; sem normalização','limite_teste':args.limite,
            'status':'matriz parcial' if len(series)<n_meta else 'matriz disponível; QC biológico pendente'}
    (SAIDA/'09b_resumo.json').write_text(json.dumps(resumo,indent=2,ensure_ascii=False),encoding='utf-8')
    rel=('ETAPA 09B | TCGA-PAAD | PROCESSAMENTO RNA-SEQ\n'+
         '\n'.join(f'{k}: {v}' for k,v in resumo.items())+
         '\n\nIdentificação paciente-arquivo proveniente da etapa 09 (metadados GDC).\n'
         'Amostras Primary Tumor; pacientes com múltiplos arquivos/amostras não são agregados.\n'
         'Validação de tamanho e MD5 contra manifesto; contagens STAR unstranded inteiras.\n'
         'Gene IDs preservados, inclusive versões ENSG. Não foram aplicados filtros ou normalização.\n'
         'Verificar alinhamento amostra-aliquota, versões de anotação, qualidade biológica e efeitos de lote.\n'
         'Matriz não é entrada pronta para modelagem; evitar vazamento entre treino e teste.\n'
         'Amostras ausentes/erro não são imputadas. Sem download automático por padrão.\n')
    (SAIDA/'09b_relatorio_metodologico.txt').write_text(rel,encoding='utf-8')
    print('\n'+rel+'\nResultados:',SAIDA)
    if not series:print('Nenhum arquivo válido encontrado. Use --baixar ou indique --diretorio-arquivos.')

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'ERRO ETAPA 09B: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
