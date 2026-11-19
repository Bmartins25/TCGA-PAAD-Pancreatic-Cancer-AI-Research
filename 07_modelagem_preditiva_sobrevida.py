# -*- coding: utf-8 -*-
"""07 | TCGA-PAAD - Comparacao exploratoria de modelos de sobrevivencia.

Dependencias basicas (Python 3.9):
    python -m pip install "numpy<2" pandas scipy matplotlib scikit-learn lifelines
Opcional para Random Survival Forest:
    python -m pip install scikit-survival

Execucao: python 07_modelagem_preditiva_sobrevida.py

AVISO: estudo exploratorio. A origem dos tempos, a censura e os estadios
clinicos devem ser auditados antes de inferencias ou uso clinico.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    RSF_DISPONIVEL = True
except ImportError:
    RSF_DISPONIVEL = False

BASE = Path(__file__).resolve().parent
ARQUIVO = BASE / 'pacientes_tcga_paad_exploratorio.csv'
SAIDA = BASE / 'resultados_modelagem_07'
SAIDA.mkdir(exist_ok=True)
SEMENTE = 2027
N_SPLITS = 5
N_REPETICOES = 3
COL = {
    'id':'cases.submitter_id', 'status':'demographic.vital_status',
    'death':'demographic.days_to_death',
    'follow':'diagnoses.days_to_last_follow_up',
    'age':'diagnoses.age_at_diagnosis',
    'stage':'diagnoses.ajcc_pathologic_stage',
}
AZUL='#203A53'; VERDE='#17877E'; LARANJA='#CB8745'; CINZA='#64748B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False})

def salvar(fig, nome):
    fig.savefig(SAIDA / nome, dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    print('Grafico:', SAIDA / nome)

def preparar():
    if not ARQUIVO.exists():
        raise FileNotFoundError(f'CSV nao encontrado: {ARQUIVO}')
    d = pd.read_csv(ARQUIVO, low_memory=False)
    faltantes = [v for v in COL.values() if v not in d.columns]
    if faltantes:
        raise ValueError(f'Colunas ausentes: {faltantes}')
    if d[COL['id']].isna().any() or d[COL['id']].duplicated().any():
        raise ValueError('IDs ausentes ou duplicados: revisar script 03.')
    for chave in ('death','follow','age'):
        d[COL[chave]] = pd.to_numeric(d[COL[chave]], errors='coerce')
    d['evento'] = d[COL['status']].astype('string').str.strip().str.lower().map({'dead':1,'alive':0})
    d['tempo_dias'] = np.where(d['evento'].eq(1), d[COL['death']], d[COL['follow']])
    d['idade_anos'] = d[COL['age']] / 365.25
    mapa = {'STAGE IA':'I','STAGE IB':'I','STAGE IIA':'II','STAGE IIB':'II',
            'STAGE III':'III-IV','STAGE IV':'III-IV'}
    d['grupo_estagio'] = d[COL['stage']].astype('string').str.strip().str.upper().map(mapa)
    valido = (d['evento'].notna() & d['tempo_dias'].gt(0) &
              d['idade_anos'].between(18,110) & d['grupo_estagio'].notna())
    d.loc[~valido, [COL['id'],'evento','tempo_dias','idade_anos','grupo_estagio']].to_csv(
        SAIDA/'07_exclusoes.csv', index=False, encoding='utf-8-sig')
    a = d.loc[valido].copy().reset_index(drop=True)
    a['idade_10anos'] = (a['idade_anos'] - 65) / 10
    a['estagio_II'] = (a['grupo_estagio'] == 'II').astype(int)
    a['estagio_III_IV'] = (a['grupo_estagio'] == 'III-IV').astype(int)
    if a['evento'].nunique() != 2:
        raise ValueError('Sao necessarios eventos e censuras.')
    return d, a

VARIAVEIS = ['idade_10anos','estagio_II','estagio_III_IV']

def c_index(tempo, evento, risco):
    # lifelines espera score crescente com maior sobrevida; risco tem sinal oposto.
    return float(concordance_index(np.asarray(tempo,float),
                                   -np.asarray(risco,float),
                                   np.asarray(evento,int)))

def ajustar_cox(treino, teste, penalizer=0.0, l1_ratio=0.0):
    modelo = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    modelo.fit(treino[['tempo_dias','evento']+VARIAVEIS],
               duration_col='tempo_dias', event_col='evento', show_progress=False)
    risco = modelo.predict_partial_hazard(teste[VARIAVEIS]).to_numpy().ravel()
    return risco

def executar_cv(a):
    x = a[VARIAVEIS].copy()
    estratos = a['evento'].astype(int).to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPETICOES,
                                 random_state=SEMENTE)
    resultados=[]; predicoes=[]; falhas=[]
    modelos = ['Cox tradicional','Cox ridge (0.1)','Cox elastic net (0.1)']
    if RSF_DISPONIVEL:
        modelos.append('Random Survival Forest')
    for k,(idx_tr,idx_te) in enumerate(cv.split(x,estratos), start=1):
        tr=a.iloc[idx_tr].copy(); te=a.iloc[idx_te].copy()
        repeticao=(k-1)//N_SPLITS+1
        dobra=(k-1)%N_SPLITS+1
        # Nenhum ajuste e realizado sobre a dobra de teste.
        for nome in modelos:
            try:
                if nome=='Cox tradicional':
                    risco=ajustar_cox(tr,te,0.0,0.0)
                elif nome=='Cox ridge (0.1)':
                    risco=ajustar_cox(tr,te,0.1,0.0)
                elif nome=='Cox elastic net (0.1)':
                    risco=ajustar_cox(tr,te,0.1,0.5)
                else:
                    y_tr=Surv.from_arrays(event=tr['evento'].astype(bool).to_numpy(),
                                          time=tr['tempo_dias'].to_numpy(float))
                    modelo=RandomSurvivalForest(n_estimators=200,min_samples_split=10,
                          min_samples_leaf=5,max_features='sqrt',random_state=SEMENTE+k,
                          n_jobs=-1)
                    modelo.fit(tr[VARIAVEIS].to_numpy(float),y_tr)
                    risco=modelo.predict(te[VARIAVEIS].to_numpy(float))
                if not np.isfinite(risco).all():
                    raise ValueError('Predicoes nao finitas')
                valor=c_index(te['tempo_dias'],te['evento'],risco)
                resultados.append({'modelo':nome,'repeticao':repeticao,'dobra':dobra,
                                   'n_treino':len(tr),'n_teste':len(te),
                                   'eventos_teste':int(te['evento'].sum()),
                                   'c_index':valor})
                for i, r in zip(idx_te, risco):
                    predicoes.append({'modelo':nome,'repeticao':repeticao,'dobra':dobra,
                                     'id_paciente':a.iloc[i][COL['id']],
                                     'tempo_dias':float(a.iloc[i]['tempo_dias']),
                                     'evento':int(a.iloc[i]['evento']),
                                     'escore_risco':float(r)})
                print(f'CV {k:02d}/{N_SPLITS*N_REPETICOES} | {nome:25s} | C-index={valor:.3f}')
            except Exception as e:
                falhas.append({'modelo':nome,'repeticao':repeticao,'dobra':dobra,
                               'erro':f'{type(e).__name__}: {e}'})
                print(f'FALHA CV {k:02d} | {nome}: {e}')
    pd.DataFrame(falhas,columns=['modelo','repeticao','dobra','erro']).to_csv(
        SAIDA/'07_falhas.csv',index=False,encoding='utf-8-sig')
    if not resultados:
        raise RuntimeError('Nenhum ajuste valido. Consultar 07_falhas.csv.')
    res=pd.DataFrame(resultados)
    res.to_csv(SAIDA/'07_resultados_por_dobra.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predicoes).to_csv(SAIDA/'07_predicoes_fora_da_amostra.csv',
                                   index=False,encoding='utf-8-sig')
    resumo=(res.groupby('modelo',sort=False)['c_index']
            .agg(n_dobras='count',media='mean',desvio_padrao='std',mediana='median',
                 minimo='min',maximo='max').reset_index())
    resumo['n_dobras_esperadas']=N_SPLITS*N_REPETICOES
    resumo['completo']=resumo['n_dobras'].eq(N_SPLITS*N_REPETICOES)
    resumo.to_csv(SAIDA/'07_resumo_modelos.csv',index=False,encoding='utf-8-sig')
    return res,resumo,falhas,modelos

def graficos(res,resumo):
    ordem=resumo.sort_values('media',ascending=False)['modelo'].tolist()
    cores={'Cox tradicional':AZUL,'Cox ridge (0.1)':VERDE,
           'Cox elastic net (0.1)':LARANJA,'Random Survival Forest':'#B95D67'}
    fig,ax=plt.subplots(figsize=(11,5.8))
    pos=np.arange(len(ordem))
    valores=[res.loc[res.modelo==m,'c_index'].to_numpy() for m in ordem]
    b=ax.boxplot(valores,positions=pos,patch_artist=True,widths=.48,showfliers=True)
    for patch,nome in zip(b['boxes'],ordem):
        patch.set_facecolor(cores[nome]);patch.set_alpha(.3)
    for j,(nome,v) in enumerate(zip(ordem,valores)):
        jitter=np.random.default_rng(SEMENTE+j).uniform(-.09,.09,len(v))
        ax.scatter(j+jitter,v,color=cores[nome],s=24,alpha=.65,zorder=3)
    ax.axhline(.5,color=CINZA,linestyle='--',linewidth=1.2,label='Discriminacao aleatoria (0,5)')
    ax.set_xticks(pos);ax.set_xticklabels(ordem,rotation=12,ha='right')
    ax.set_ylabel('C-index na dobra de teste')
    ax.set_title('TCGA-PAAD | Discriminacao fora da amostra',fontweight='bold',color=AZUL)
    ax.grid(axis='y',color='#E5EBF0');ax.legend(loc='lower right')
    fig.text(.5,.005,'Dobras repetidas nao sao observacoes independentes; dispersao apenas descritiva.',
             ha='center',fontsize=9,color=CINZA)
    fig.tight_layout(rect=[0,.035,1,1]);salvar(fig,'07_comparacao_cindex.png')
    fig,ax=plt.subplots(figsize=(10,5.4))
    for nome in ordem:
        s=res[res.modelo==nome].sort_values(['repeticao','dobra'])
        ax.plot(np.arange(1,len(s)+1),s.c_index,marker='o',markersize=4,
                linewidth=1.5,alpha=.8,label=nome,color=cores[nome])
    ax.axhline(.5,color=CINZA,ls='--',lw=1)
    ax.set_xlabel('Dobra avaliada (ordenada por repeticao)')
    ax.set_ylabel('C-index na dobra de teste')
    ax.set_title('Variabilidade da discriminacao entre dobras',fontweight='bold',color=AZUL)
    ax.grid(color='#E5EBF0');ax.legend(fontsize=9)
    fig.tight_layout();salvar(fig,'07_variabilidade_dobras.png')

def relatorio(d,a,resumo,falhas,modelos):
    linhas=[
        'SCRIPT 07 | TCGA-PAAD | RELATORIO EXPLORATORIO',
        f'Coorte original: {len(d)} | elegiveis: {len(a)} | eventos: {int(a.evento.sum())} | censuras: {int((1-a.evento).sum())}',
        'Unidade de analise: paciente unico; desfecho: obito (1) vs censura (0).',
        'Tempo: days_to_death se falecido; days_to_last_follow_up se vivo.',
        'Covariaveis: idade (10 anos, centrada em 65) e estagios II / III-IV (referencia I).',
        f'Validacao: RepeatedStratifiedKFold, {N_SPLITS} dobras x {N_REPETICOES} repeticoes, seed {SEMENTE}.',
        'Estratificacao por evento; todos os ajustes feitos apenas nos dados de treinamento.',
        'Cox sem penalizacao; Cox ridge penalizer=0.1; Cox elastic net penalizer=0.1, l1_ratio=0.5.',
        f'Random Survival Forest: {"executado" if RSF_DISPONIVEL else "NAO executado: scikit-survival nao instalado"}.',
        'Metrica: C-index em cada dobra de teste; valores agregados sao DESCRITIVOS.',
        'Dobras repetidas sao dependentes; DP entre dobras nao e intervalo de confianca.',
        'Hiperparametros predefinidos, SEM busca ou selecao otimizada por CV aninhada.',
        'Nao foram estimados Brier Score, AUC dependente do tempo nem calibracao.',
        'Nao interpretar a ordenacao por media como superioridade estatistica.',
        'Nenhuma modalidade omica foi usada: somente tres covariaveis clinicas.',
        'Limitacoes: grupos III-IV pequenos; possivel instabilidade de estimativas.',
        'Auditar origem temporal, censura, codificacao de estadios 0/0a e vies de selecao.',
        'Resultados nao representam validacao externa, clinica, causal ou ferramenta assistencial.',
        f'Modelos solicitados: {", ".join(modelos)}',
        f'Falhas registradas: {len(falhas)}',
        '', 'RESUMO:', resumo.to_string(index=False,float_format=lambda v:f'{v:.4f}')
    ]
    (SAIDA/'07_relatorio_metodologico.txt').write_text('\n'.join(linhas),encoding='utf-8')
    (SAIDA/'07_configuracao.json').write_text(json.dumps({
        'semente':SEMENTE,'n_splits':N_SPLITS,'n_repeticoes':N_REPETICOES,
        'rsf_disponivel':RSF_DISPONIVEL,'covariaveis':VARIAVEIS,
        'modelos':modelos},ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'+'\n'.join(linhas))

def main():
    d,a=preparar()
    print(f'Coorte: {len(d)} | elegiveis: {len(a)} | eventos: {int(a.evento.sum())}')
    if not RSF_DISPONIVEL:
        print('AVISO: scikit-survival nao instalado; RSF sera omitido, sem substituicao ficticia.')
    res,resumo,falhas,modelos=executar_cv(a)
    graficos(res,resumo)
    relatorio(d,a,resumo,falhas,modelos)
    print('\nArquivos salvos em:',SAIDA)

if __name__=='__main__':
    main()
