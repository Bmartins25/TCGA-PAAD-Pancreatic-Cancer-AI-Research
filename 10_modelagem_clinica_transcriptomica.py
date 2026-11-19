#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETAPA 10 — comparação exploratória Cox: clínico, RNA-seq e integrado.
Dependências: pip install numpy pandas scikit-learn lifelines matplotlib
Não constitui validação clínica. Filtragem/seleção RNA são feitas no treino de cada dobra.
"""
from pathlib import Path
import argparse, json, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

BASE = Path(__file__).resolve().parent

def stage_group(value):
    s = str(value).upper().replace('STAGE','').strip()
    if s in ('', 'NAN', 'NONE', 'NOT REPORTED', 'UNKNOWN', 'STAGE X', 'X'): return np.nan
    if s.startswith('IV'): return 'IV'
    if s.startswith('III'): return 'III'
    if s.startswith('II'): return 'II'
    if s.startswith('I'): return 'I'
    return np.nan

def cindex(ytime, ye, risk):
    if len(np.unique(ye)) < 2: return np.nan
    return float(concordance_index(ytime, -risk, ye))

def design_clinical(train, test):
    a = train[['idade_anos']].copy(); b = test[['idade_anos']].copy()
    med = a['idade_anos'].median()
    if not np.isfinite(med): raise ValueError('Idade indisponível no treino.')
    a['idade_anos'] = a['idade_anos'].fillna(med)
    b['idade_anos'] = b['idade_anos'].fillna(med)
    cats = ['II','III','IV']
    for c in cats:
        a['estagio_'+c] = (train['estagio'] == c).astype(float)
        b['estagio_'+c] = (test['estagio'] == c).astype(float)
    return a, b

def design_rna(raw_train, raw_test, n_genes, min_cpm, min_fraction):
    # genes x pacientes, contagens inteiras; treino controla TODAS as decisões de filtragem/seleção
    lib_tr = raw_train.sum(axis=0)
    lib_te = raw_test.sum(axis=0)
    if (lib_tr <= 0).any() or (lib_te <= 0).any():
        raise ValueError('Biblioteca de tamanho zero.')
    cpm_tr = raw_train / lib_tr[None,:] * 1e6
    detected = (cpm_tr >= min_cpm).sum(axis=1)
    keep = detected >= max(2, int(np.ceil(min_fraction * raw_train.shape[1])))
    if not np.any(keep): raise ValueError('Nenhum gene passou pelo filtro do treino.')
    # logCPM apenas por profundidade de biblioteca individual; seleção por variância no treino
    logtr = np.log2(1 + cpm_tr[keep])
    var = np.var(logtr, axis=1)
    eligible = np.where(keep)[0]
    order = np.argsort(-var, kind='stable')[:min(n_genes, len(var))]
    chosen = eligible[order]
    xtr = np.log2(1 + raw_train[chosen] / lib_tr[None,:] * 1e6).T
    xte = np.log2(1 + raw_test[chosen] / lib_te[None,:] * 1e6).T
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xte = scaler.transform(xte)
    cols = ['gene_'+str(i) for i in chosen]
    return pd.DataFrame(xtr, columns=cols), pd.DataFrame(xte, columns=cols), chosen

def fit_predict(xtr, xte, train, penalizer):
    # remove colunas constantes exclusivamente com base no treino
    cols = xtr.columns[xtr.std(axis=0, ddof=0).fillna(0).gt(1e-9)]
    if len(cols) == 0: raise ValueError('Sem covariáveis variáveis no treino.')
    d = xtr.loc[:, cols].reset_index(drop=True).copy()
    d['tempo'] = train['tempo_dias'].to_numpy(float)
    d['evento'] = train['evento_obito'].to_numpy(int)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(d, duration_col='tempo', event_col='evento', show_progress=False)
        return model.predict_log_partial_hazard(xte.loc[:, cols]).to_numpy().ravel()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=BASE/'resultados_integracao_09d')
    ap.add_argument('--saida', type=Path, default=BASE/'resultados_modelagem_10')
    ap.add_argument('--genes', type=int, default=20, help='Máximo de genes selecionados por variância no treino')
    ap.add_argument('--dobras', type=int, default=3)
    ap.add_argument('--repeticoes', type=int, default=2)
    ap.add_argument('--penalizacao', type=float, default=1.0)
    ap.add_argument('--min-cpm', type=float, default=1.0)
    ap.add_argument('--fracao-minima', type=float, default=0.20)
    args = ap.parse_args()
    if args.genes < 1 or args.dobras < 2 or args.repeticoes < 1 or args.penalizacao <= 0:
        ap.error('Use genes >=1, dobras >=2, repeticoes >=1 e penalizacao >0.')
    args.saida.mkdir(parents=True, exist_ok=True)
    clin_path = args.base/'09d_pacientes_pareados_clinicos.csv'
    mat_path = args.base/'09d_contagens_brutas_pareadas.csv.gz'
    for p in (clin_path, mat_path):
        if not p.is_file(): raise FileNotFoundError(f'Ausente: {p}. Execute a 09D com --exportar-matriz.')
    cli = pd.read_csv(clin_path, low_memory=False)
    required = ['id_paciente','evento_obito','tempo_dias','diagnoses.age_at_diagnosis','diagnoses.ajcc_pathologic_stage']
    missing = [c for c in required if c not in cli.columns]
    if missing: raise ValueError(f'Colunas clínicas ausentes: {missing}')
    cli['id_paciente'] = cli.id_paciente.astype(str).str.strip().str.upper()
    if cli.id_paciente.duplicated().any(): raise ValueError('IDs clínicos duplicados.')
    cli['evento_obito'] = pd.to_numeric(cli.evento_obito, errors='coerce')
    cli['tempo_dias'] = pd.to_numeric(cli.tempo_dias, errors='coerce')
    # TCGA age_at_diagnosis é tipicamente em DIAS, não anos
    cli['idade_anos'] = pd.to_numeric(cli['diagnoses.age_at_diagnosis'], errors='coerce') / 365.25
    cli['estagio'] = cli['diagnoses.ajcc_pathologic_stage'].map(stage_group)
    cli = cli.loc[cli.evento_obito.isin([0,1]) & cli.tempo_dias.gt(0)].copy()
    cli = cli.loc[cli.idade_anos.between(18,110) & cli.estagio.isin(['I','II','III','IV'])].copy()
    if cli.empty: raise ValueError('Sem pacientes elegíveis; auditar unidade de idade e estágio.')
    print('ETAPA 10 | Lendo matriz bruta; aguarde...', flush=True)
    mat = pd.read_csv(mat_path, index_col=0)
    mat.columns = mat.columns.astype(str).str.strip().str.upper()
    if mat.columns.duplicated().any() or mat.index.duplicated().any(): raise ValueError('IDs duplicados na matriz.')
    ids = sorted(set(cli.id_paciente).intersection(mat.columns))
    cli = cli.set_index('id_paciente').loc[ids].reset_index()
    if len(ids) < args.dobras*4: raise ValueError('Coorte insuficiente para a validação proposta.')
    if cli.evento_obito.value_counts().min() < args.dobras: raise ValueError('Poucos eventos ou censuras para estratificação.')
    mat = mat.loc[:, ids]
    if not np.issubdtype(mat.to_numpy().dtype, np.number): raise ValueError('Contagens não numéricas.')
    arr = mat.to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all() or (arr < 0).any() or not np.allclose(arr, np.rint(arr)):
        raise ValueError('Matriz não contém contagens brutas inteiras não negativas.')
    cv = RepeatedStratifiedKFold(n_splits=args.dobras, n_repeats=args.repeticoes, random_state=2027)
    records, predictions, selections, failures = [], [], [], []
    models = ['clinico','transcriptomico','integrado']
    for fold,(itr,ite) in enumerate(cv.split(np.zeros(len(cli)),cli.evento_obito.astype(int)),1):
        tr = cli.iloc[itr].reset_index(drop=True)
        te = cli.iloc[ite].reset_index(drop=True)
        clinical_tr,clinical_te = design_clinical(tr,te)
        try:
            rna_tr,rna_te,genes = design_rna(arr[:,itr],arr[:,ite],args.genes,args.min_cpm,args.fracao_minima)
            for g in genes: selections.append({'dobra':fold,'gene_id':str(mat.index[g])})
        except Exception as exc:
            failures.append({'dobra':fold,'modelo':'RNA','erro':repr(exc)})
            rna_tr=rna_te=None
        for model in models:
            if model != 'clinico' and rna_tr is None: continue
            if model == 'clinico': xtr,xte = clinical_tr,clinical_te
            elif model == 'transcriptomico': xtr,xte = rna_tr,rna_te
            else:
                xtr = pd.concat([clinical_tr,rna_tr],axis=1)
                xte = pd.concat([clinical_te,rna_te],axis=1)
            try:
                risk = fit_predict(xtr,xte,tr,args.penalizacao)
                score = cindex(te.tempo_dias.to_numpy(float),te.evento_obito.to_numpy(int),risk)
                records.append({'dobra':fold,'modelo':model,'n_treino':len(tr),'n_teste':len(te),
                                'eventos_teste':int(te.evento_obito.sum()),'c_index':score})
                for i,r in enumerate(risk):
                    predictions.append({'dobra':fold,'modelo':model,'id_paciente':te.id_paciente.iloc[i],
                                        'tempo_dias':te.tempo_dias.iloc[i],'evento_obito':int(te.evento_obito.iloc[i]),
                                        'risco_log_relativo':float(r)})
                print(f'Dobra {fold}/{args.dobras*args.repeticoes} | {model}: C-index={score:.3f}',flush=True)
            except Exception as exc:
                failures.append({'dobra':fold,'modelo':model,'erro':repr(exc)})
                print(f'AVISO dobra {fold} {model}: {exc}',flush=True)
    res = pd.DataFrame(records,columns=['dobra','modelo','n_treino','n_teste','eventos_teste','c_index'])
    res.to_csv(args.saida/'10_resultados_por_dobra.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(args.saida/'10_predicoes_fora_amostra.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(selections).to_csv(args.saida/'10_genes_selecionados_por_dobra.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(failures,columns=['dobra','modelo','erro']).to_csv(args.saida/'10_falhas.csv',index=False,encoding='utf-8-sig')
    summary = res.groupby('modelo').agg(n_dobras=('c_index','count'),c_index_medio=('c_index','mean'),
                                        c_index_dp=('c_index','std'),c_index_mediana=('c_index','median')).reset_index() if len(res) else pd.DataFrame()
    summary.to_csv(args.saida/'10_resumo_modelos.csv',index=False,encoding='utf-8-sig')
    cfg = {'pacientes_elegiveis':len(cli),'eventos':int(cli.evento_obito.sum()),'censuras':int((cli.evento_obito==0).sum()),
           'genes_brutos':len(mat),'modelos':models,'dobras':args.dobras,'repeticoes':args.repeticoes,
           'max_genes_por_dobra':args.genes,'penalizacao':args.penalizacao,'min_cpm':args.min_cpm,
           'fracao_minima':args.fracao_minima,'seed':2027,'falhas':len(failures)}
    (args.saida/'10_configuracao.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    report = ('ETAPA 10 | TCGA-PAAD | COMPARAÇÃO EXPLORATÓRIA\n\n'+json.dumps(cfg,ensure_ascii=False,indent=2)+
              '\n\nRESULTADOS\n'+summary.to_string(index=False)+
              '\n\nLIMITAÇÕES\nAUC(t), Brier(t) e calibração não calculados nesta versão; ver etapa 08.\n'
              'Validação cruzada repetida NÃO equivale a validação externa nem fornece IC independentes.\n'
              'Filtros de expressão, seleção por variância e escala são ajustados dentro de cada treino.\n'
              'CPM é exploratório; não substitui normalização apropriada para análise diferencial.\n'
              'Idade convertida de dias para anos; estágios ausentes/exóticos excluídos.\n'
              'Não há ajuste por lote, verificação de origem temporal, subtipos ou qualidade biológica.\n'
              'Não foi realizada busca de hiperparâmetros; parâmetros são pré-definidos.\n'
              'Sem inferência causal, superioridade estatística, validade clínica ou generalização externa.\n')
    (args.saida/'10_relatorio_metodologico.txt').write_text(report,encoding='utf-8')
    print('\n'+report+'\nArquivos em:',args.saida)
    if failures or len(res) != args.dobras*args.repeticoes*3:
        print('ATENÇÃO: há falhas ou resultados incompletos; conferir 10_falhas.csv.',file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(f'ERRO ETAPA 10: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
