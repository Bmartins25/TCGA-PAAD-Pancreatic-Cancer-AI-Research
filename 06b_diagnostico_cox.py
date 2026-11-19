# -*- coding: utf-8 -*-
"""06B | Diagnosticos exploratorios do modelo de Cox TCGA-PAAD.
Requer: python -m pip install numpy pandas scipy matplotlib statsmodels
Execute ao lado de pacientes_tcga_paad_exploratorio.csv.
Nao altera arquivos originais. Teste de Schoenfeld abaixo e triagem aproximada,
NAO o teste formal de Grambsch-Therneau; use lifelines para confirmacao formal.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.duration.hazard_regression import PHReg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'pacientes_tcga_paad_exploratorio.csv'
OUT=ROOT/'resultados_cox_06b';OUT.mkdir(exist_ok=True)
NAVY='#203A53';TEAL='#16877C';ORANGE='#CB8745';RED='#B95D67';GRAY='#637487'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
COL={'id':'cases.submitter_id','status':'demographic.vital_status','death':'demographic.days_to_death','follow':'diagnoses.days_to_last_follow_up','age':'diagnoses.age_at_diagnosis','stage':'diagnoses.ajcc_pathologic_stage'}

def save(fig,name):
    fig.savefig(OUT/name,dpi=300,bbox_inches='tight',facecolor='white',pad_inches=.25)
    plt.close(fig)
    print('Gráfico:',OUT/name)

def fit_cox(df,columns):
    x=df[columns].to_numpy(float)
    result=PHReg(df.tempo_dias.to_numpy(float),x,status=df.evento.to_numpy(int),ties='efron').fit(disp=0)
    beta=np.asarray(result.params);se=np.asarray(result.bse)
    if not (np.isfinite(beta).all() and np.isfinite(se).all()):
        raise ValueError('Estimativas nao finitas')
    return result,pd.DataFrame({'variavel':columns,'HR':np.exp(beta),'IC95_inferior':np.exp(beta-1.96*se),'IC95_superior':np.exp(beta+1.96*se),'p_valor':result.pvalues})

def main():
    if not SRC.exists():raise FileNotFoundError('CSV nao encontrado: '+str(SRC))
    d=pd.read_csv(SRC)
    missing=[v for v in COL.values() if v not in d.columns]
    if missing:raise ValueError('Colunas ausentes: '+str(missing))
    if d[COL['id']].isna().any() or d[COL['id']].duplicated().any():raise ValueError('IDs ausentes ou repetidos')
    for c in ('death','follow','age'):d[COL[c]]=pd.to_numeric(d[COL[c]],errors='coerce')
    d['evento']=d[COL['status']].astype('string').str.strip().str.lower().map({'dead':1,'alive':0})
    d['tempo_dias']=np.where(d.evento==1,d[COL['death']],d[COL['follow']])
    d['idade_anos']=d[COL['age']]/365.25
    mapping={'STAGE IA':'I','STAGE IB':'I','STAGE IIA':'II','STAGE IIB':'II','STAGE III':'III-IV','STAGE IV':'III-IV'}
    d['grupo_estagio']=d[COL['stage']].astype('string').str.strip().str.upper().map(mapping)
    valid=d.evento.notna()&d.tempo_dias.gt(0)&d.idade_anos.between(18,110)&d.grupo_estagio.notna()
    a=d.loc[valid].copy().reset_index(drop=True)
    a['idade_10anos']=(a.idade_anos-65)/10
    a['estagio_II']=(a.grupo_estagio=='II').astype(int)
    a['estagio_III_IV']=(a.grupo_estagio=='III-IV').astype(int)
    cols=['idade_10anos','estagio_II','estagio_III_IV']
    fit,tab=fit_cox(a,cols)
    tab.to_csv(OUT/'06b_cox_principal.csv',index=False,encoding='utf-8-sig')
    # Residuos de Schoenfeld, apenas eventos: tendencia linear em log(tempo)
    # P-valores de regressao simples NAO substituem teste formal PH de Grambsch-Therneau.
    scho=np.asarray(fit.schoenfeld_residuals)
    ev=(a.evento.to_numpy()==1)&np.isfinite(scho).all(axis=1)
    xx=np.log1p(a.loc[ev,'tempo_dias'].to_numpy(float))
    fig,axes=plt.subplots(1,3,figsize=(15.6,4.8),sharex=True)
    names=['Idade (10 anos)','Estágio II vs. I','Estágio III–IV vs. I']
    colors=[TEAL,ORANGE,RED]; rows=[]
    for j,ax in enumerate(axes):
        yy=scho[ev,j]
        lr=linregress(xx,yy)
        ax.scatter(xx,yy,s=24,color=colors[j],alpha=.52,edgecolors='none')
        z=np.linspace(xx.min(),xx.max(),100)
        ax.plot(z,lr.intercept+lr.slope*z,color=NAVY,lw=2.2)
        ax.axhline(0,color=GRAY,ls='--',lw=1)
        ax.set_title(names[j],fontweight='bold',color=NAVY,fontsize=12)
        ax.set_xlabel('ln(1 + tempo em dias)')
        ax.grid(axis='y',color='#E7EDF0',lw=.8)
        rows.append({'variavel':cols[j],'slope':lr.slope,'p_tendencia_aproximada':lr.pvalue,'n_eventos':int(ev.sum())})
    axes[0].set_ylabel('Resíduo de Schoenfeld (não escalonado)')
    fig.suptitle('Diagnóstico exploratório de proporcionalidade dos riscos',fontweight='bold',fontsize=17,color=NAVY,y=1.05)
    fig.text(.5,-.035,'Triagem por tendência linear: não equivale ao teste formal de Grambsch–Therneau.',ha='center',fontsize=10,color=GRAY)
    fig.tight_layout();save(fig,'06b_residuos_schoenfeld.png')
    pd.DataFrame(rows).to_csv(OUT/'06b_triagem_schoenfeld.csv',index=False,encoding='utf-8-sig')
    # Martingale versus idade: LOWESS de statsmodels, apenas inspeção visual de forma funcional
    from statsmodels.nonparametric.smoothers_lowess import lowess
    mart=np.asarray(fit.martingale_residuals)
    finite=np.isfinite(mart)&np.isfinite(a.idade_anos.to_numpy())
    age=a.loc[finite,'idade_anos'].to_numpy(float);res=mart[finite]
    smooth=lowess(res,age,frac=.45,it=0,return_sorted=True)
    fig,ax=plt.subplots(figsize=(10.6,5.7))
    ax.scatter(age,res,color=TEAL,alpha=.38,s=30,edgecolors='none',label='Pacientes')
    ax.plot(smooth[:,0],smooth[:,1],color=RED,lw=3,label='LOWESS (exploratório)')
    ax.axhline(0,color=GRAY,ls='--',lw=1)
    ax.set(xlabel='Idade ao diagnóstico (anos)',ylabel='Resíduo de Martingale',title='Forma funcional da idade — diagnóstico visual')
    ax.title.set_fontweight('bold');ax.title.set_color(NAVY)
    ax.grid(axis='y',color='#E7EDF0');ax.legend(frameon=False)
    fig.text(.12,-.015,'Curvatura persistente pode indicar necessidade de splines; inspeção não é teste formal.',fontsize=9,color=GRAY)
    fig.tight_layout();save(fig,'06b_martingale_idade.png')
    # Sensibilidade: agrupamento binário II-IV vs I, e remoção de III-IV
    sens=[]
    for label,sub,features in [
        ('Principal: I / II / III-IV',a,cols),
        ('Binário: II-IV vs I',a.assign(estagio_avancado=(a.grupo_estagio!='I').astype(int)),['idade_10anos','estagio_avancado']),
        ('Restrito: apenas I e II',a.loc[a.grupo_estagio.isin(['I','II'])].copy(),['idade_10anos','estagio_II'])]:
        try:
            f,t=fit_cox(sub,features)
            for _,r in t.iterrows():sens.append({'cenario':label,'n':len(sub),'eventos':int(sub.evento.sum()),**r.to_dict()})
        except Exception as exc:
            sens.append({'cenario':label,'n':len(sub),'eventos':int(sub.evento.sum()),'variavel':'FALHA: '+str(exc)})
    s=pd.DataFrame(sens)
    s.to_csv(OUT/'06b_sensibilidade.csv',index=False,encoding='utf-8-sig')
    age_s=s.loc[s.variavel=='idade_10anos'].copy()
    fig,ax=plt.subplots(figsize=(10.5,5.3))
    ys=np.arange(len(age_s))[::-1]
    for y,(_,r) in zip(ys,age_s.iterrows()):
        ax.errorbar(r.HR,y,xerr=[[r.HR-r.IC95_inferior],[r.IC95_superior-r.HR]],fmt='o',color=TEAL,capsize=5,markersize=9,lw=2.2)
        ax.text(r.IC95_superior+.025,y,f'{r.HR:.2f} [{r.IC95_inferior:.2f}; {r.IC95_superior:.2f}]',va='center',fontsize=10,color=NAVY)
    ax.axvline(1,ls='--',color=GRAY)
    ax.set_yticks(ys,age_s.cenario.tolist());ax.tick_params(axis='y',length=0,pad=10)
    ax.set_xlim(max(0,min(age_s.IC95_inferior)-.2),max(age_s.IC95_superior)+.8)
    ax.set_xlabel('HR por 10 anos de idade · IC 95%')
    ax.set_title('Sensibilidade da associação entre idade e sobrevida',fontweight='bold',color=NAVY,pad=18)
    ax.grid(axis='x',color='#E7EDF0');fig.tight_layout();save(fig,'06b_sensibilidade_idade.png')
    report=(f'Coorte original: {len(d)}; elegíveis: {len(a)}; eventos: {int(a.evento.sum())}; excluídos: {len(d)-len(a)}.\n'
            'Método: PHReg, empates Efron; mesma codificação do Script 06.\n'
            'Schoenfeld: tendência linear dos resíduos não escalonados contra log1p(tempo), triagem exploratória.\n'
            'NÃO é teste formal de Grambsch-Therneau. P-valores não são prova de proporcionalidade.\n'
            'Martingale: inspeção LOWESS; não é teste formal de linearidade.\n'
            'Sensibilidade: diferentes agrupamentos de estágios; grupos pequenos e confusão residual limitam conclusões.\n'
            'Revisar codificação clínica, origem temporal, censura e pressupostos antes de inferência definitiva.\n')
    (OUT/'06b_relatorio_metodologico.txt').write_text(report,encoding='utf-8')
    print('\nSCRIPT 06B FINALIZADO');print(report);print('Arquivos em:',OUT)
if __name__=='__main__':main()
