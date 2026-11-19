#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCGA-PAAD | Etapa 09C: QC, filtragem e normalização exploratória de RNA-seq.
Python 3.9+. Dependências: pandas numpy matplotlib scipy (scipy opcional).
Entrada: resultados_rnaseq_09b/09b_matriz_contagens_brutas.csv.gz
Saída: resultados_rnaseq_09c/
ATENÇÃO: CPM/log2(CPM+1) serve para exploração e visualização, NÃO é uma
normalização inferencial para expressão diferencial; parâmetros de filtragem,
seleção de genes e transformações usadas em ML devem ser ajustados no treino.
"""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ENTRADA = BASE / 'resultados_rnaseq_09b' / '09b_matriz_contagens_brutas.csv.gz'
SAIDA = BASE / 'resultados_rnaseq_09c'
AZUL, VERDE, LARANJA, CINZA = '#203A53', '#17877E', '#CB8745', '#64748B'

def figura(nome):
    plt.savefig(SAIDA / nome, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def estilo():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titlesize':14,'axes.titleweight':'bold',
                         'figure.facecolor':'white','axes.labelcolor':AZUL,
                         'text.color':AZUL,'axes.titlecolor':AZUL})

def main():
    global SAIDA
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entrada', type=Path, default=ENTRADA)
    p.add_argument('--saida', type=Path, default=SAIDA)
    p.add_argument('--cpm-min', type=float, default=1.0, help='CPM mínimo para presença do gene')
    p.add_argument('--fracao-min', type=float, default=0.10, help='Fração mínima de amostras com CPM suficiente')
    p.add_argument('--genes-pca', type=int, default=2000, help='Genes mais variáveis para PCA exploratória')
    p.add_argument('--sem-matriz-log', action='store_true', help='Não exportar matriz log2(CPM+1)')
    args = p.parse_args()
    SAIDA = args.saida.resolve()
    SAIDA.mkdir(parents=True, exist_ok=True)
    estilo()
    if not args.entrada.is_file():
        raise FileNotFoundError(f'Matriz 09B não encontrada: {args.entrada}\nExecute 09B ou use --entrada CAMINHO')
    if not (0 < args.fracao_min <= 1) or args.cpm_min < 0 or args.genes_pca < 2:
        raise ValueError('Parâmetros inválidos: fracao-min em (0,1], cpm-min >= 0, genes-pca >= 2')
    print('09C | Lendo matriz bruta...', flush=True)
    df = pd.read_csv(args.entrada, compression='infer', index_col=0)
    if df.empty or df.shape[1] < 3:
        raise ValueError('Matriz vazia ou com menos de três amostras')
    if df.index.has_duplicates or df.columns.has_duplicates:
        raise ValueError('Gene IDs ou pacientes duplicados; interrompido para evitar pareamento incorreto')
    if not df.index.astype(str).str.match(r'^ENSG\d+(?:\.\d+)?$').all():
        raise ValueError('IDs gênicos inesperados; esperado ENSG com versão opcional')
    dados = df.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(dados).all() or (dados < 0).any() or not np.allclose(dados, np.rint(dados)):
        raise ValueError('Contagens ausentes, negativas, infinitas ou fracionárias')
    if not np.all(dados <= np.iinfo(np.int64).max):
        raise ValueError('Contagem excede limite de inteiro 64-bit')
    dados = dados.astype(np.int64)
    pacientes = df.columns.astype(str)
    genes = df.index.astype(str)
    total = dados.sum(axis=0, dtype=np.int64)
    detectados = (dados > 0).sum(axis=0)
    if (total == 0).any():
        raise ValueError('Amostras com biblioteca de tamanho zero: ' + ', '.join(pacientes[total == 0]))
    n_amostras = len(pacientes)
    print(f'Genes: {len(genes)} | pacientes: {n_amostras}', flush=True)
    qc = pd.DataFrame({'id_paciente': pacientes, 'total_contagens':total,
                       'genes_detectados':detectados,
                       'percentual_genes_detectados':100*detectados/len(genes),
                       'log10_total_contagens':np.log10(total)})
    # Regras explícitas, sem exclusão automática de pacientes por valores extremos.
    for campo in ('total_contagens','genes_detectados'):
        v = qc[campo].to_numpy(dtype=float)
        q1,q3 = np.percentile(v,[25,75]); iqr=q3-q1
        qc['alerta_iqr_'+campo] = (v < q1-1.5*iqr) | (v > q3+1.5*iqr)
    qc['alerta_exploratorio'] = qc.filter(like='alerta_iqr_').any(axis=1)
    qc.to_csv(SAIDA/'09c_qc_amostras.csv',index=False,encoding='utf-8-sig')

    # CPM = contagem / soma de contagens da amostra * 1e6.
    # Matriz original permanece intacta. Nenhuma correção de lote ou imputação.
    cpm = dados.astype(np.float64) / total[np.newaxis,:] * 1e6
    minimo_amostras = max(2,int(np.ceil(args.fracao_min*n_amostras)))
    presente = (cpm >= args.cpm_min).sum(axis=1)
    manter = presente >= minimo_amostras
    n_filtrados = int(manter.sum())
    if n_filtrados < 3:
        raise ValueError('Menos de 3 genes após filtragem. Reveja parâmetros sem sobrescrever matriz bruta.')
    filtragem = pd.DataFrame({'gene_id':genes, 'n_amostras_cpm_min':presente,
                              'manter_filtro_exploratorio':manter,
                              'soma_contagens':dados.sum(axis=1,dtype=np.int64)})
    filtragem.to_csv(SAIDA/'09c_auditoria_filtragem_genes.csv.gz',index=False,compression='gzip')
    pd.DataFrame(dados[manter,:],index=genes[manter],columns=pacientes).rename_axis('gene_id').to_csv(
        SAIDA/'09c_contagens_brutas_filtradas_exploratorio.csv.gz',compression='gzip')
    log_cpm = np.log2(cpm[manter,:]+1.0)
    if not args.sem_matriz_log:
        pd.DataFrame(log_cpm,index=genes[manter],columns=pacientes).rename_axis('gene_id').to_csv(
            SAIDA/'09c_log2_cpm_exploratorio.csv.gz',compression='gzip',float_format='%.6f')
    print(f'Genes após filtro exploratório: {n_filtrados}',flush=True)

    # Figura 1: distribuição de profundidade e detecção por amostra.
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    ax[0].hist(qc.log10_total_contagens,bins=22,color=AZUL,edgecolor='white')
    ax[0].set(xlabel='log10(contagens totais)',ylabel='Amostras',title='Profundidade de biblioteca')
    ax[1].scatter(qc.total_contagens/1e6,qc.genes_detectados,s=32,color=VERDE,alpha=.75)
    ax[1].set(xlabel='Contagens totais (milhões)',ylabel='Genes com contagem > 0',title='Complexidade observada')
    for a in ax:a.grid(alpha=.16);a.set_axisbelow(True)
    fig.suptitle('TCGA-PAAD | Qualidade técnica por amostra',fontsize=16,fontweight='bold',color=AZUL)
    fig.tight_layout();figura('09c_01_qualidade_bibliotecas.png')

    # Figura 2: genes antes/depois do filtro.
    fig,ax=plt.subplots(figsize=(8,4.6))
    labels=['Genes originais','Genes mantidos','Genes removidos']
    valores=[len(genes),n_filtrados,len(genes)-n_filtrados]
    bars=ax.barh(labels[::-1],valores[::-1],color=[LARANJA,VERDE,AZUL],height=.55)
    for bar in bars:ax.text(bar.get_width()+max(valores)*.015,bar.get_y()+bar.get_height()/2,f'{int(bar.get_width()):,}'.replace(',','.'),va='center')
    ax.set_xlim(0,max(valores)*1.2);ax.set_xlabel('Número de genes');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('TCGA-PAAD | Filtragem exploratória por expressão')
    fig.tight_layout();figura('09c_02_filtragem_genes.png')

    # Figura 3: distribuições log CPM por amostra (amostras em ordem de profundidade).
    ordem=np.argsort(total)
    pos=np.unique(np.linspace(0,n_amostras-1,min(35,n_amostras),dtype=int))
    amostras_plot=ordem[pos]
    fig,ax=plt.subplots(figsize=(13,5.3))
    ax.boxplot([log_cpm[:,j] for j in amostras_plot],showfliers=False,
               patch_artist=True,boxprops={'facecolor':'#B8DCD7','edgecolor':VERDE},
               medianprops={'color':AZUL,'linewidth':1.3},
               whiskerprops={'color':CINZA},capprops={'color':CINZA})
    ax.set(xlabel='Amostras ordenadas por profundidade (subconjunto para visualização)',ylabel='log2(CPM + 1)')
    ax.set_title('TCGA-PAAD | Distribuição de expressão após transformação')
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.tight_layout();figura('09c_03_distribuicao_logcpm.png')

    # PCA exploratória: genes mais variáveis escolhidos usando a coorte completa;
    # esta PCA NÃO pode ser usada como transformação em validação fora da amostra.
    variancias=log_cpm.var(axis=1)
    n_pca=min(args.genes_pca,len(variancias))
    indices=np.argsort(variancias)[-n_pca:]
    X=log_cpm[indices,:].T.copy()
    X-=X.mean(axis=0,keepdims=True)
    # SVD de amostras x genes (pequena matriz: ~178x2000).
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    pcs=U[:,:2]*S[:2]
    variacao=(S[:2]**2)/(S**2).sum()*100 if np.any(S) else np.array([0.,0.])
    pca=pd.DataFrame({'id_paciente':pacientes,'PC1':pcs[:,0], 'PC2':pcs[:,1],
                      'total_contagens':total,'alerta_exploratorio':qc.alerta_exploratorio})
    pca.to_csv(SAIDA/'09c_pca_amostras.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(9.2,6.6))
    scatter=ax.scatter(pcs[:,0],pcs[:,1],c=np.log10(total),cmap='viridis',s=44,alpha=.83,edgecolors='white',linewidths=.3)
    fig.colorbar(scatter,ax=ax,label='log10(contagens totais)')
    ax.set(xlabel=f'PC1 ({variacao[0]:.1f}% da variância)',ylabel=f'PC2 ({variacao[1]:.1f}% da variância)')
    ax.set_title('TCGA-PAAD | PCA exploratória de expressão gênica')
    ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.01,'PCA ajustada em toda a coorte para QC; não utilizar como componente de validação preditiva.',ha='center',fontsize=9,color=CINZA)
    fig.tight_layout(rect=[0,.04,1,1]);figura('09c_04_pca_exploratoria.png')

    # Checagem de correspondência com etapa 09B sem assumir dados clínicos completos.
    validos_path=BASE/'resultados_rnaseq_09b'/'09b_pacientes_amostras_validos.csv'
    mapeamento={'arquivo_encontrado':validos_path.is_file(),'pacientes_iguais':None,'diferencas':[]}
    if validos_path.is_file():
        mapa=pd.read_csv(validos_path,dtype=str)
        if 'id_paciente' in mapa.columns:
            ids=set(mapa.id_paciente.dropna().astype(str))
            dif=sorted(set(pacientes).symmetric_difference(ids))
            mapeamento.update(pacientes_iguais=len(dif)==0,diferencas=dif[:30])
            if dif:print(f'ATENÇÃO: {len(dif)} divergências entre matriz e mapa de pacientes')

    resumo={'etapa':'09C','status':'QC exploratório concluído; revisão biológica pendente',
            'entrada':str(args.entrada.resolve()),'pacientes':n_amostras,
            'genes_originais':int(len(genes)),'genes_mantidos':n_filtrados,
            'genes_removidos':int(len(genes)-n_filtrados),
            'cpm_min':args.cpm_min,'fracao_min':args.fracao_min,'amostras_minimas':minimo_amostras,
            'amostras_com_alerta_iqr':int(qc.alerta_exploratorio.sum()),
            'genes_utilizados_pca':n_pca,'variancia_pc1_percentual':round(float(variacao[0]),3),
            'variancia_pc2_percentual':round(float(variacao[1]),3),
            'mapeamento_09b':mapeamento,
            'normalizacao':'CPM e log2(CPM+1), exploratória; sem ajuste de tamanho/composição por TMM/DESeq2',
            'efeitos_lote':'não avaliados: requer metadados técnicos',
            'amostras_excluidas_automaticamente':0,
            'validacao_preditiva':'não realizada; refazer seleção e transformações dentro de cada dobra de treino'}
    (SAIDA/'09c_resumo.json').write_text(json.dumps(resumo,indent=2,ensure_ascii=False),encoding='utf-8')
    rel=(
        'ETAPA 09C | TCGA-PAAD | CONTROLE DE QUALIDADE E NORMALIZAÇÃO EXPLORATÓRIA\n\n'
        + '\n'.join(f'{k}: {v}' for k,v in resumo.items())
        + '\n\nLIMITAÇÕES E CONDUTAS OBRIGATÓRIAS:\n'
        '1. Dados: contagens STAR unstranded; verificar strandedness, anotação GENCODE e versões ENSG.\n'
        '2. CPM usa soma de contagens de cada biblioteca; log2(CPM+1) é exploratório.\n'
        '3. Filtro: CPM >= limiar em >= fração de amostras; não constitui seleção para inferência clínica.\n'
        '4. Alertas IQR são triagem, não autorizam exclusão automática.\n'
        '5. PCA e seleção de genes mais variáveis foram ajustadas na coorte completa APENAS para QC.\n'
        '6. Antes de modelagem, dividir por paciente e ajustar filtragem, escala e PCA só no treino de cada dobra.\n'
        '7. Não foi realizada correção de lote; requer metadados de centro, placa, workflow e/ou processamento.\n'
        '8. Para expressão diferencial, usar métodos próprios para contagens (DESeq2/edgeR) e desenho apropriado.\n'
        '9. Conferir composição celular, pureza tumoral, amostra/alíquota, dados clínicos e potenciais confundidores.\n'
        '10. Não foram realizadas validação externa, avaliação clínica ou inferência causal.\n'
    )
    (SAIDA/'09c_relatorio_metodologico.txt').write_text(rel,encoding='utf-8')
    print('\n'+rel+'\nResultados:',SAIDA)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'ERRO 09C: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
