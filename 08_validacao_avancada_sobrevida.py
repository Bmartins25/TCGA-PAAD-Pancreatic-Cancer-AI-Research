# -*- coding: utf-8 -*-
"""ETAPA 08 | TCGA-PAAD | Validacao interna de sobrevida.
Python 3.9+: pip install numpy pandas matplotlib scikit-learn lifelines
Coloque este script, 07_modelagem_preditiva_sobrevida.py e
pacientes_tcga_paad_exploratorio.csv na mesma pasta.
Execucao: python 08_validacao_avancada_sobrevida.py

AVISO: resultados exploratorios; nao representam validacao externa ou clinica.
AUC(t) e Brier(t) usam IPCW por Kaplan-Meier da censura no TREINO.
Nao usar as predicoes da etapa 07 como probabilidades de sobrevivencia.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

BASE=Path(__file__).resolve().parent
CSV=BASE/"pacientes_tcga_paad_exploratorio.csv"
OUT=BASE/"resultados_validacao_08"
OUT.mkdir(exist_ok=True)
SEED=2027
SPLITS=5
REPS=3
HORIZONTES=[365,730,1095]
VAR=["idade_10anos","estagio_II","estagio_III_IV"]
MODELOS={
    "Cox tradicional":(0.0,0.0),
    "Cox ridge (0.1)":(0.1,0.0),
    "Cox elastic net (0.1)":(0.1,0.5)
}
COL={
"id":"cases.submitter_id","status":"demographic.vital_status",
"death":"demographic.days_to_death","follow":"diagnoses.days_to_last_follow_up",
"age":"diagnoses.age_at_diagnosis","stage":"diagnoses.ajcc_pathologic_stage"
}
CORES={"Cox tradicional":"#203A53","Cox ridge (0.1)":"#17877E",
       "Cox elastic net (0.1)":"#CB8745"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
"axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})

def salvar(fig,nome):
    fig.savefig(OUT/nome,dpi=300,bbox_inches="tight",pad_inches=.22)
    plt.close(fig)
    print("Grafico:",OUT/nome)

def preparar():
    if not CSV.is_file():
        raise FileNotFoundError("Coloque pacientes_tcga_paad_exploratorio.csv junto ao script.")
    d=pd.read_csv(CSV,low_memory=False)
    faltam=[v for v in COL.values() if v not in d.columns]
    if faltam: raise ValueError("Colunas ausentes: "+str(faltam))
    if d[COL["id"]].isna().any() or d[COL["id"]].duplicated().any():
        raise ValueError("IDs ausentes/duplicados; rever consolidacao.")
    for c in ("death","follow","age"):
        d[COL[c]]=pd.to_numeric(d[COL[c]],errors="coerce")
    d["evento"]=d[COL["status"]].astype("string").str.strip().str.lower().map({"dead":1,"alive":0})
    d["tempo_dias"]=np.where(d["evento"].eq(1),d[COL["death"]],d[COL["follow"]])
    d["idade_anos"]=d[COL["age"]]/365.25
    mapa={"STAGE IA":"I","STAGE IB":"I","STAGE IIA":"II","STAGE IIB":"II",
          "STAGE III":"III-IV","STAGE IV":"III-IV"}
    d["grupo_estagio"]=d[COL["stage"]].astype("string").str.strip().str.upper().map(mapa)
    ok=(d.evento.notna()&d.tempo_dias.gt(0)&
        d.idade_anos.between(18,110)&d.grupo_estagio.notna())
    a=d.loc[ok].copy().reset_index(drop=True)
    a["idade_10anos"]=(a.idade_anos-65)/10
    a["estagio_II"]=(a.grupo_estagio=="II").astype(int)
    a["estagio_III_IV"]=(a.grupo_estagio=="III-IV").astype(int)
    if len(a)<30 or a.evento.nunique()!=2: raise ValueError("Coorte insuficiente.")
    return d,a

def km_censura(treino):
    # evento=1 significa censura observada (evento clinico=0)
    km=KaplanMeierFitter()
    km.fit(treino.tempo_dias.to_numpy(float),
           event_observed=1-treino.evento.to_numpy(int))
    return km

def g_antes(km,t):
    # G(t-) para peso de evento; G(t) para observados alem do horizonte.
    tempos=km.survival_function_.index.to_numpy(float)
    probs=km.survival_function_.iloc[:,0].to_numpy(float)
    ix=np.searchsorted(tempos,np.asarray(t,dtype=float),side="left")-1
    return np.where(ix>=0,probs[np.maximum(ix,0)],1.0)

def g_ate(km,t):
    tempos=km.survival_function_.index.to_numpy(float)
    probs=km.survival_function_.iloc[:,0].to_numpy(float)
    ix=np.searchsorted(tempos,np.asarray(t,dtype=float),side="right")-1
    return np.where(ix>=0,probs[np.maximum(ix,0)],1.0)

def pesos_ipcw(t,e,h,km):
    t=np.asarray(t,float); e=np.asarray(e,int)
    caso=(t<=h)&(e==1)
    controle=t>h
    pesos=np.zeros(len(t))
    if caso.any():
        gc=g_antes(km,t[caso])
        if np.any(gc<=0.05): return None,None,None
        pesos[caso]=1/gc
    gh=float(g_ate(km,[h])[0])
    if gh<=0.05: return None,None,None
    pesos[controle]=1/gh
    return pesos,caso,controle

def auc_ipcw(t,e,risco,h,km):
    # AUC cumulativa/dinamica; casos eventos ate h, controles sem evento ate h.
    w,caso,controle=pesos_ipcw(t,e,h,km)
    if w is None or caso.sum()<5 or controle.sum()<5: return np.nan
    rc=risco[caso]; rn=risco[controle]
    cmp=(rc[:,None]>rn[None,:]).astype(float)+0.5*(rc[:,None]==rn[None,:])
    wc=w[caso]
    # Controles recebem pesos unitarios: fator G(h) cancela no quociente AUC.
    return float((cmp*wc[:,None]).sum()/(wc.sum()*len(rn)))

def brier_ipcw(t,e,s,h,km):
    w,caso,controle=pesos_ipcw(t,e,h,km)
    if w is None or caso.sum()<5 or controle.sum()<5: return np.nan
    alvo=np.where(caso,0.,1.)
    return float(np.sum(w*(alvo-s)**2)/len(t))

def km_observada(t,e,h):
    if len(t)<8 or e.sum()<1: return np.nan
    km=KaplanMeierFitter().fit(t,event_observed=e)
    return float(km.predict(h))

def main():
    bruto,a=preparar()
    print("Coorte:",len(bruto),"| elegiveis:",len(a),
          "| eventos:",int(a.evento.sum()))
    cv=RepeatedStratifiedKFold(n_splits=SPLITS,n_repeats=REPS,random_state=SEED)
    resultados=[]; calibracao=[]; falhas=[]; predicoes=[]
    for k,(it,iv) in enumerate(cv.split(a[VAR],a.evento.astype(int)),1):
        tr=a.iloc[it].copy(); te=a.iloc[iv].copy()
        rep=(k-1)//SPLITS+1; dobra=(k-1)%SPLITS+1
        km=km_censura(tr)
        for nome,(pen,l1) in MODELOS.items():
            try:
                mod=CoxPHFitter(penalizer=pen,l1_ratio=l1)
                mod.fit(tr[["tempo_dias","evento"]+VAR],
                        duration_col="tempo_dias",event_col="evento",show_progress=False)
                risco=mod.predict_partial_hazard(te[VAR]).to_numpy(float).ravel()
                s=mod.predict_survival_function(te[VAR],times=HORIZONTES)
                if s.shape!=(len(HORIZONTES),len(te)):
                    raise RuntimeError("Dimensao inesperada das probabilidades.")
                if not np.isfinite(s.to_numpy()).all():
                    raise ValueError("Probabilidades nao finitas.")
                t=te.tempo_dias.to_numpy(float); e=te.evento.to_numpy(int)
                c=float(concordance_index(t,-risco,e))
                for j,h in enumerate(HORIZONTES):
                    # Nao extrapolar para alem do suporte do treino.
                    if h>=tr.tempo_dias.max():
                        auc=np.nan; bs=np.nan; motivo="horizonte fora do suporte de treino"
                    else:
                        auc=auc_ipcw(t,e,risco,h,km)
                        bs=brier_ipcw(t,e,s.iloc[j].to_numpy(float),h,km)
                        motivo="" if np.isfinite(auc) and np.isfinite(bs) else "IPCW/casos/controles insuficientes"
                    resultados.append({"modelo":nome,"repeticao":rep,"dobra":dobra,
                      "horizonte_dias":h,"n_treino":len(tr),"n_teste":len(te),
                      "eventos_teste":int(e.sum()),"c_index":c,
                      "auc_ipcw":auc,"brier_ipcw":bs,"observacao":motivo})
                    for i,(_,pac) in enumerate(te.iterrows()):
                        predicoes.append({"modelo":nome,"repeticao":rep,"dobra":dobra,
                          "id_paciente":pac[COL["id"]],"horizonte_dias":h,
                          "tempo_dias":t[i],"evento":e[i],
                          "prob_sobrevida":float(s.iloc[j,i]),"risco_relativo":risco[i]})
                    # Calibracao em 2 grupos por dobra; KM observado, nao taxa bruta.
                    # A avaliacao e descritiva: grupos pequenos, censura e repeticoes.
                    if np.isfinite(bs):
                        estim=s.iloc[j].to_numpy(float)
                        med=np.median(estim)
                        grupos=np.where(estim<=med,"Sobrevida prevista menor","Sobrevida prevista maior")
                        for grupo in np.unique(grupos):
                            ix=grupos==grupo
                            obs=km_observada(t[ix],e[ix],h)
                            if np.isfinite(obs):
                                calibracao.append({"modelo":nome,"repeticao":rep,
                                  "dobra":dobra,"horizonte_dias":h,"grupo":grupo,
                                  "n":int(ix.sum()),"prevista_media":float(estim[ix].mean()),
                                  "observada_km":obs})
                print(f"CV {k:02d}/15 | {nome:24s} | C-index={c:.3f}")
            except Exception as ex:
                falhas.append({"modelo":nome,"repeticao":rep,"dobra":dobra,
                               "erro":type(ex).__name__+": "+str(ex)})
                print("FALHA:",nome,"rep",rep,"dobra",dobra,ex)
    res=pd.DataFrame(resultados)
    res.to_csv(OUT/"08_metricas_por_dobra.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(falhas,columns=["modelo","repeticao","dobra","erro"]).to_csv(
        OUT/"08_falhas.csv",index=False,encoding="utf-8-sig")
    if res.empty: raise RuntimeError("Nenhum modelo ajustado. Consulte 08_falhas.csv.")
    pred=pd.DataFrame(predicoes)
    pred.to_csv(OUT/"08_predicoes_probabilidades.csv",index=False,encoding="utf-8-sig")
    cal=pd.DataFrame(calibracao)
    cal.to_csv(OUT/"08_calibracao_por_dobra.csv",index=False,encoding="utf-8-sig")
    resumo=(res.groupby(["modelo","horizonte_dias"],sort=False)
      .agg(n_dobras=("c_index","count"),n_auc=("auc_ipcw","count"),
           auc_media=("auc_ipcw","mean"),auc_dp=("auc_ipcw","std"),
           n_brier=("brier_ipcw","count"),brier_medio=("brier_ipcw","mean"),
           brier_dp=("brier_ipcw","std")).reset_index())
    resumo.to_csv(OUT/"08_resumo_metricas.csv",index=False,encoding="utf-8-sig")
    for metrica,arquivo,titulo,ylabel in [
        ("auc_ipcw","08_auc_tempo.png","Discriminacao dependente do tempo","AUC(t) - IPCW"),
        ("brier_ipcw","08_brier_tempo.png","Erro de predicao de sobrevida","Brier Score(t) - IPCW")]:
        fig,ax=plt.subplots(figsize=(10,5.5))
        for nome in MODELOS:
            s=res[res.modelo==nome].groupby("horizonte_dias")[metrica].agg(["mean","std","count"])
            ax.errorbar(s.index/365.25,s["mean"],yerr=s["std"],label=nome,
                marker="o",capsize=4,lw=2,color=CORES[nome])
        if metrica=="auc_ipcw":
            ax.axhline(.5,color="#64748B",ls="--",lw=1,label="Referencia 0,5")
        ax.set_xticks(np.array(HORIZONTES)/365.25,["12 meses","24 meses","36 meses"])
        ax.set_ylabel(ylabel);ax.set_xlabel("Horizonte de avaliacao")
        ax.set_title("TCGA-PAAD | "+titulo,color="#203A53",weight="bold")
        ax.grid(axis="y",color="#E5EBF0");ax.legend(fontsize=9)
        fig.text(.5,.005,"Barras: desvio-padrao descritivo entre dobras dependentes; nao sao IC 95%.",
                 ha="center",fontsize=9,color="#64748B")
        fig.tight_layout(rect=[0,.04,1,1]);salvar(fig,arquivo)
    if not cal.empty:
        fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
        for ax,h in zip(axes,HORIZONTES):
            sub=cal[cal.horizonte_dias==h]
            for nome in MODELOS:
                z=sub[sub.modelo==nome]
                if len(z):
                    ax.scatter(z.prevista_media,z.observada_km,s=25,alpha=.35,
                               color=CORES[nome],label=nome)
            ax.plot([0,1],[0,1],ls="--",color="#64748B",lw=1)
            ax.set_title(f"{round(h/365.25*12)} meses",weight="bold")
            ax.grid(color="#E5EBF0")
            ax.set_xlabel("Sobrevida prevista")
        axes[0].set_ylabel("Sobrevida observada (KM por grupo)")
        axes[0].legend(fontsize=8,loc="lower right")
        fig.suptitle("TCGA-PAAD | Calibracao exploratoria por dobra",weight="bold",
                     color="#203A53")
        fig.text(.5,.005,"Pontos por grupo e dobra; nao equivalem a curva de calibracao validada.",
                 ha="center",fontsize=9,color="#64748B")
        fig.tight_layout(rect=[0,.04,1,.94]);salvar(fig,"08_calibracao_exploratoria.png")
    configuracao={"semente":SEED,"dobras":SPLITS,"repeticoes":REPS,
      "horizontes_dias":HORIZONTES,"modelos":MODELOS,
      "censura":"KM ajustada somente no treino para IPCW",
      "calibracao":"KM por 2 grupos em cada dobra; exploratoria"}
    (OUT/"08_configuracao.json").write_text(
        json.dumps(configuracao,ensure_ascii=False,indent=2),encoding="utf-8")
    linhas=["ETAPA 08 | TCGA-PAAD | RELATORIO METODOLOGICO",
      f"Original: {len(bruto)} | elegiveis: {len(a)} | eventos: {int(a.evento.sum())}",
      "Mesmo preparo, variaveis e divisao de CV da etapa 07 (seed 2027).",
      "AUC(t): cumulativa/dinamica com pesos IPCW para casos.",
      "Brier(t): pesos IPCW de Kaplan-Meier da censura estimada NO TREINO.",
      "Predicao de sobrevivencia: CoxPHFitter.predict_survival_function.",
      "Calibracao: 2 grupos por dobra, observado por Kaplan-Meier; exploratoria.",
      "Horizontes sem suporte ou pesos instaveis retornam NaN, nao sao imputados.",
      "Hiperparametros predefinidos; sem selecao aninhada ou validacao externa.",
      "Desvios-padrao entre dobras repetidas nao representam IC independentes.",
      "Nao calculado IBS integrado, calibracao suavizada ou IC bootstrap.",
      "Verificar origem temporal, censura, estadios 0/0a e vies de selecao.",
      "Sem conclusoes clinicas ou de superioridade estatistica.",
      f"Falhas de ajuste: {len(falhas)}","",
      resumo.to_string(index=False,float_format=lambda x:f"{x:.4f}")]
    (OUT/"08_relatorio_metodologico.txt").write_text("\n".join(linhas),encoding="utf-8")
    print("\n"+"\n".join(linhas))
    print("\nArquivos em:",OUT)

if __name__=="__main__":
    main()
