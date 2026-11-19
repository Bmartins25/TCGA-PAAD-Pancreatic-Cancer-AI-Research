#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etapa 09 | TCGA-PAAD: auditoria de pareamento clínico–RNA-seq.
Não treina modelos nem presume que arquivos de manifesto já foram baixados.
Dependências: pip install pandas numpy matplotlib requests
"""
from pathlib import Path
import argparse, json, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CLINICO = BASE / 'pacientes_tcga_paad_exploratorio.csv'
SAIDA = BASE / 'resultados_integracao_09'
COL_ID = 'cases.submitter_id'
CORES = {'azul':'#203A53', 'verde':'#17877E', 'laranja':'#CB8745', 'cinza':'#64748B'}

def gravar(df, nome):
    df.to_csv(SAIDA / nome, index=False, encoding='utf-8-sig')

def ler_manifesto(caminho):
    df = pd.read_csv(caminho, sep='\t', dtype=str)
    req = {'id', 'filename'}
    if not req.issubset(df.columns):
        raise ValueError(f'Manifesto inválido: {caminho}; colunas esperadas: {req}')
    return df

def buscar_metadados(ids, tamanho_lote=80):
    """Consulta API pública do GDC; sem fabricar vínculo paciente–arquivo."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError('Instale requests: python -m pip install requests') from exc
    url = 'https://api.gdc.cancer.gov/files'
    saida = []
    for i in range(0, len(ids), tamanho_lote):
        lote = ids[i:i+tamanho_lote]
        filtros = {'op':'in', 'content':{'field':'files.file_id', 'value':lote}}
        params = {'filters':json.dumps(filtros), 'fields':
                  'file_id,file_name,data_type,experimental_strategy,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type,cases.samples.sample_type_id',
                  'format':'JSON','size':len(lote)}
        r = requests.get(url, params=params, timeout=75)
        r.raise_for_status()
        hits = r.json().get('data', {}).get('hits', [])
        saida.extend(hits)
        print(f'Metadados GDC: {min(i+tamanho_lote,len(ids))}/{len(ids)} IDs consultados')
    return saida

def achatar(hits):
    linhas=[]
    for h in hits:
        casos=h.get('cases') or []
        if not casos:
            linhas.append({'id':h.get('file_id'),'id_paciente':None,'id_amostra':None,
                           'tipo_amostra':None,'arquivo_gdc':h.get('file_name'), 'tipo_dado':h.get('data_type')})
        for caso in casos:
            amostras=caso.get('samples') or [None]
            for amostra in amostras:
                amostra=amostra or {}
                linhas.append({'id':h.get('file_id'),'id_paciente':caso.get('submitter_id'),
                               'id_amostra':amostra.get('submitter_id'),
                               'tipo_amostra':amostra.get('sample_type'),
                               'arquivo_gdc':h.get('file_name'),'tipo_dado':h.get('data_type')})
    return pd.DataFrame(linhas, columns=['id','id_paciente','id_amostra','tipo_amostra','arquivo_gdc','tipo_dado'])

def figura_contagens(n_clinico,n_com_rna,n_sem_rna,n_arquivos):
    fig,ax=plt.subplots(figsize=(10.8,5.6))
    nomes=['Pacientes clínicos','Com RNA-seq confirmado','Sem RNA-seq confirmado','Arquivos no manifesto']
    vals=[n_clinico,n_com_rna,n_sem_rna,n_arquivos]
    barras=ax.barh(nomes[::-1],vals[::-1],color=[CORES['cinza'],CORES['laranja'],CORES['verde'],CORES['azul']][::-1],height=.55)
    for b in barras:
        ax.text(b.get_width()+max(vals)*.012,b.get_y()+b.get_height()/2,f'{int(b.get_width())}',va='center',fontsize=11)
    ax.set_xlim(0,max(vals)*1.20 if max(vals) else 1)
    ax.set_xlabel('Contagem (pacientes ou arquivos, conforme categoria)')
    ax.set_title('TCGA-PAAD | Auditoria de integração clínico–RNA-seq',fontweight='bold',color=CORES['azul'],pad=20)
    ax.grid(axis='x',alpha=.16); ax.set_axisbelow(True)
    for sp in ('top','right','left'): ax.spines[sp].set_visible(False)
    fig.text(.12,.015,'Vínculos confirmados exclusivamente por metadados do GDC; arquivos não são pacientes.',fontsize=9,color=CORES['cinza'])
    fig.subplots_adjust(left=.29,right=.93,top=.86,bottom=.17)
    fig.savefig(SAIDA/'09_fluxo_pareamento.png',dpi=300,facecolor='white'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clinico',type=Path,default=CLINICO)
    ap.add_argument('--manifesto',type=Path,default=None)
    ap.add_argument('--metadados',type=Path,default=None,help='JSON de hits da API GDC salvo previamente')
    ap.add_argument('--consultar-gdc',action='store_true',help='Consulta online a API do GDC')
    args=ap.parse_args()
    SAIDA.mkdir(exist_ok=True)
    if not args.clinico.exists(): raise FileNotFoundError(f'Arquivo clínico ausente: {args.clinico}')
    clin=pd.read_csv(args.clinico,dtype={COL_ID:str})
    if COL_ID not in clin: raise ValueError(f'Coluna clínica ausente: {COL_ID}')
    clin['id_paciente']=clin[COL_ID].astype('string').str.strip().str.upper()
    if clin['id_paciente'].isna().any() or clin['id_paciente'].duplicated().any():
        raise ValueError('IDs clínicos ausentes/duplicados; revisar a consolidação antes de integrar')
    if not clin['id_paciente'].str.fullmatch(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}').all():
        raise ValueError('IDs clínicos fora do padrão TCGA de paciente; revisar')
    manifestos=sorted(BASE.glob('gdc_manifest*.txt'))
    if args.manifesto: manifestos=[args.manifesto]
    if not manifestos: raise FileNotFoundError('Nenhum gdc_manifest*.txt encontrado na pasta do script')
    listas=[ler_manifesto(p) for p in manifestos]
    man=pd.concat(listas,ignore_index=True).drop_duplicates(subset=['id'])
    man=man[man['filename'].str.contains('rna_seq|gene_counts',case=False,na=False)].copy()
    if man.empty: raise ValueError('Nenhum arquivo RNA-seq identificado nos manifestos')
    gravar(man,'09_inventario_manifesto.csv')
    print(f'Pacientes clínicos: {len(clin)} | arquivos RNA-seq no manifesto: {len(man)}')
    if args.metadados:
        bruto=json.loads(args.metadados.read_text(encoding='utf-8'))
        hits=bruto.get('data',{}).get('hits',[]) if isinstance(bruto,dict) else bruto
    elif args.consultar_gdc:
        hits=buscar_metadados(man['id'].dropna().unique().tolist())
        (SAIDA/'09_metadados_gdc.json').write_text(json.dumps(hits,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        hits=[]
    if not hits:
        gravar(pd.DataFrame(columns=['id','id_paciente','id_amostra','tipo_amostra','arquivo_gdc','tipo_dado']), '09_mapeamento_gdc.csv')
        rel=('ETAPA 09 — AUDITORIA INICIAL\n'
             f'Pacientes clínicos: {len(clin)}\nArquivos no manifesto: {len(man)}\n'
             'PAREAMENTO NÃO REALIZADO: metadados GDC ausentes.\n'
             'Execute novamente com --consultar-gdc, ou forneça --metadados JSON.\n'
             'Não é permitido inferir pacientes a partir de nomes UUID de arquivos.\n')
        (SAIDA/'09_relatorio_metodologico.txt').write_text(rel,encoding='utf-8')
        print(rel)
        return
    mapa=achatar(hits)
    mapa['id_paciente']=mapa['id_paciente'].astype('string').str.upper().str.strip()
    gravar(mapa,'09_mapeamento_gdc.csv')
    ids_man=set(man['id'].astype(str)); ids_meta=set(mapa['id'].dropna().astype(str))
    sem_meta=ids_man-ids_meta
    if sem_meta: print(f'ATENÇÃO: {len(sem_meta)} IDs do manifesto sem retorno na API')
    mapa=mapa[mapa['id'].isin(ids_man)].copy()
    mapa['em_coorte_clinica']=mapa['id_paciente'].isin(clin['id_paciente'])
    mapa['amostra_tumor_primario']=mapa['tipo_amostra'].fillna('').str.lower().eq('primary tumor')
    gravar(mapa,'09_auditoria_amostras.csv')
    confirmado=mapa[mapa['em_coorte_clinica'] & mapa['amostra_tumor_primario']].copy()
    cont=confirmado.groupby('id_paciente').agg(n_arquivos=('id','nunique'),n_amostras=('id_amostra','nunique')).reset_index()
    unidos=clin.merge(cont,on='id_paciente',how='left',validate='one_to_one')
    unidos['n_arquivos']=unidos['n_arquivos'].fillna(0).astype(int)
    unidos['n_amostras']=unidos['n_amostras'].fillna(0).astype(int)
    unidos['rna_seq_tumor_confirmado']=unidos['n_arquivos']>0
    gravar(unidos,'09_coorte_clinica_com_disponibilidade_rnaseq.csv')
    gravar(confirmado,'09_pares_clinico_rnaseq.csv')
    duplicados=cont[(cont['n_arquivos']>1)|(cont['n_amostras']>1)]
    gravar(duplicados,'09_pacientes_multiplos_arquivos.csv')
    n_sim=int(unidos['rna_seq_tumor_confirmado'].sum()); n_nao=len(unidos)-n_sim
    figura_contagens(len(clin),n_sim,n_nao,len(man))
    resumo={'pacientes_clinicos':len(clin),'arquivos_manifesto':len(man),
            'arquivos_com_metadados':len(ids_man & ids_meta),
            'pacientes_com_rna_tumor_primario':n_sim,'pacientes_sem_rna_tumor_primario':n_nao,
            'pacientes_com_multiplos_arquivos_ou_amostras':len(duplicados),
            'status':'auditoria de metadados; expressão gênica ainda não integrada'}
    (SAIDA/'09_resumo.json').write_text(json.dumps(resumo,indent=2,ensure_ascii=False),encoding='utf-8')
    rel=('ETAPA 09 — TCGA-PAAD | AUDITORIA DE METADADOS RNA-SEQ\n'
         + '\n'.join(f'{k}: {v}' for k,v in resumo.items())
         + '\n\nCRITÉRIOS: identificador de paciente TCGA do GDC; amostra Primary Tumor.\n'
         'Arquivos múltiplos por paciente não são agregados automaticamente.\n'
         'O manifesto lista arquivos, não contém contagens gênicas nem vínculo paciente.\n'
         'Ainda NÃO foi construída matriz de expressão ou realizado controle de qualidade RNA-seq.\n'
         'Não assumir que metadados GDC garantem existência de arquivos baixados.\n'
         'Próxima fase: baixar arquivos STAR counts, conferir checksums, selecionar amostras,\n'
         'verificar tipo de quantificação, gene IDs, normalização e vazamento de informação.\n')
    (SAIDA/'09_relatorio_metodologico.txt').write_text(rel,encoding='utf-8')
    print(rel); print(f'Resultados: {SAIDA}')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERRO ETAPA 09: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
