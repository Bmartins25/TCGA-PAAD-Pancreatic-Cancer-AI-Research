# -*- coding: utf-8 -*-
"""Script 06C | TCGA-PAAD: diagnosticos formais e refinamento do Cox.
Instalacao: python -m pip install "numpy<2" pandas scipy matplotlib lifelines patsy
Executar na pasta que contem pacientes_tcga_paad_exploratorio.csv.
Resultados exploratorios, NAO validacao clinica ou causal.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2
from patsy import dmatrix, build_design_matrices
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

BASE=Path(__file__).resolve().parent
ARQUIVO=BASE/'pacientes_tcga_paad_exploratorio.csv'
SAIDA=BASE/'resultados_cox_06c'
SAIDA.mkdir(exist_ok=True)
COL={'id':'cases.submitter_id','status':'demographic.vital_status',
     'death':'demographic.days_to_death','follow':'diagnoses.days_to_last_follow_up',
     'age':'diagnoses.age_at_diagnosis','stage':'diagnoses.ajcc_pathologic_stage'}
AZUL='#203A53'; VERDE='#17877E'; VERMELHO='#B95D67'; LARANJA='#CB8745'; CINZA='#64748B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.facecolor':'white'})

def salvar(fig,nome):
    fig.savefig(SAIDA/nome,dpi=300,bbox_inches='tight',pad_inches=.25)
    plt.close(fig)
    print('Gráfico:',SAIDA/nome)

def preparar():
    if not ARQUIVO.is_file():
        raise FileNotFoundError(f'Arquivo ausente: {ARQUIVO}')
    d=pd.read_csv(ARQUIVO,low_memory=False)
    ausentes=[v for v in COL.values() if v not in d]
    if ausentes:raise ValueError('Colunas obrigatorias ausentes: '+str(ausentes))
    if d[COL['id']].isna().any() or d[COL['id']].duplicated().any():
        raise ValueError('Identificadores ausentes ou duplicados; revisar consolidacao.')
    for c in ('death','follow','age'):
        d[COL[c]]=pd.to_numeric(d[COL[c]],errors='coerce')
    d['evento']=d[COL['status']].astype('string').str.strip().str.lower().map({'dead':1,'alive':0})
    d['tempo_dias']=np.where(d.evento==1,d[COL['death']],d[COL['follow']])
    d['idade_anos']=d[COL['age']]/365.25
    mapa={'STAGE IA':'I','STAGE IB':'I','STAGE IIA':'II','STAGE IIB':'II',
          'STAGE III':'III-IV','STAGE IV':'III-IV'}
    d['grupo_estagio']=d[COL['stage']].astype('string').str.strip().str.upper().map(mapa)
    ok=d.evento.notna() & d.tempo_dias.gt(0) & d.idade_anos.between(18,110) & d.grupo_estagio.notna()
    exclusoes=d.loc[~ok,[COL['id'],'evento','tempo_dias','idade_anos','grupo_estagio']].copy()
    exclusoes.to_csv(SAIDA/'06c_exclusoes.csv',index=False,encoding='utf-8-sig')
    a=d.loc[ok].copy().reset_index(drop=True)
    a['idade_10anos']=(a.idade_anos-65)/10
    a['estagio_II']=(a.grupo_estagio=='II').astype(int)
    a['estagio_III_IV']=(a.grupo_estagio=='III-IV').astype(int)
    if a.evento.sum()<10:raise ValueError('Eventos insuficientes para diagnosticos.')
    return d,a

def ajustar(df,variaveis):
    cph=CoxPHFitter(penalizer=0.0)
    cph.fit(df[['tempo_dias','evento']+variaveis],duration_col='tempo_dias',event_col='evento',show_progress=False)
    return cph

def painel_ph(modelo,df,cols):
    # lifelines: teste baseado em residuos de Schoenfeld escalonados.
    testes={}
    for transformacao in ('rank','km'):
        resultado=proportional_hazard_test(modelo,df[['tempo_dias','evento']+cols],time_transform=transformacao)
        tab=resultado.summary.reset_index().rename(columns={'index':'variavel'})
        tab['transformacao']=transformacao
        testes[transformacao]=tab
    tabela=pd.concat(testes.values(),ignore_index=True)
    tabela.to_csv(SAIDA/'06c_testes_ph.csv',index=False,encoding='utf-8-sig')
    residuos=modelo.compute_residuals(df[['tempo_dias','evento']+cols],kind='scaled_schoenfeld')
    # Indices de residuos preservam os indices dos eventos no dataframe de ajuste.
    t=df.loc[residuos.index,'tempo_dias'].to_numpy(float)
    fig,axs=plt.subplots(1,len(cols),figsize=(15.5,5.2))
    axs=np.atleast_1d(axs)
    nomes=['Idade (10 anos)','Estágio II vs. I','Estágio III–IV vs. I']
    cores=[VERDE,LARANJA,VERMELHO]
    for j,(ax,c) in enumerate(zip(axs,cols)):
        x=np.log1p(t); y=residuos[c].to_numpy(float)
        valid=np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[valid],y[valid],s=24,alpha=.55,color=cores[j],edgecolor='none')
        if valid.sum()>=8 and np.ptp(x[valid])>0:
            # LOWESS ilustrativo, nao e o teste formal.
            from statsmodels.nonparametric.smoothers_lowess import lowess
            from scipy.stats import linregress
            curva=lowess(y[valid],x[valid],frac=.5,it=0,return_sorted=True)
            ax.plot(curva[:,0],curva[:,1],lw=2.5,color=AZUL)
        ax.axhline(0,color=CINZA,ls='--',lw=1)
        p=tabela.loc[(tabela.variavel==c)&(tabela.transformacao=='rank'),'p'].iloc[0]
        ax.set_title(f'{nomes[j]}\nTeste rank: p={p:.3g}',color=AZUL,fontweight='bold',fontsize=12)
        ax.set_xlabel('ln(1 + tempo em dias)')
        ax.grid(axis='y',color='#E5EBF0',alpha=.8)
    axs[0].set_ylabel('Resíduo de Schoenfeld escalonado')
    fig.suptitle('Proporcionalidade dos riscos | diagnóstico por covariável',fontweight='bold',color=AZUL,fontsize=17)
    fig.text(.5,.015,'LOWESS apenas visual; p-valores do teste de Schoenfeld escalonado (transformação rank).',ha='center',fontsize=10,color=CINZA)
    fig.tight_layout(rect=[0,.035,1,.93])
    salvar(fig,'06c_schoenfeld_escalonado.png')
    return tabela

def splines_idade(a,modelo_linear,cols):
    # df=4: quatro bases de spline (incluindo componente linear), comparadas a 1 termo linear.
    # knots calculados exclusivamente na coorte analitica; comparacao LRT exploratoria.
    b=dmatrix('bs(x, df=4, degree=3, include_intercept=False) - 1',
              {'x':a.idade_anos.to_numpy(float)},return_type='dataframe')
    nomes=[f'spline_idade_{i+1}' for i in range(b.shape[1])]
    design_info=b.design_info
    b.columns=nomes
    s=a[['tempo_dias','evento','estagio_II','estagio_III_IV']].copy()
    for c in nomes:s[c]=b[c].to_numpy(float)
    modelo_spline=ajustar(s,['estagio_II','estagio_III_IV']+nomes)
    gl=len(nomes)-1
    lr_raw=2*(modelo_spline.log_likelihood_-modelo_linear.log_likelihood_)
    if lr_raw < -1e-6:
        raise ValueError(f'Log-verossimilhanca spline inferior ao modelo aninhado: LRT={lr_raw:.6f}; revisar ajuste.')
    lr=max(0.,lr_raw)
    p=float(chi2.sf(lr,gl))
    comparacao=pd.DataFrame([
        {'modelo':'Idade linear','n':len(a),'eventos':int(a.evento.sum()),'parametros':len(cols),
         'log_verossimilhanca':modelo_linear.log_likelihood_,'AIC_parcial':modelo_linear.AIC_partial_,
         'concordancia_aparente':modelo_linear.concordance_index_},
        {'modelo':'Idade spline cubica (df=4)','n':len(a),'eventos':int(a.evento.sum()),'parametros':len(nomes)+2,
         'log_verossimilhanca':modelo_spline.log_likelihood_,'AIC_parcial':modelo_spline.AIC_partial_,
         'concordancia_aparente':modelo_spline.concordance_index_}])
    comparacao['LRT_spline_vs_linear']=np.nan
    comparacao['p_LRT_spline_vs_linear']=np.nan
    comparacao.loc[1,'LRT_spline_vs_linear']=lr
    comparacao.loc[1,'p_LRT_spline_vs_linear']=p
    comparacao.to_csv(SAIDA/'06c_comparacao_modelos.csv',index=False,encoding='utf-8-sig')
    # Razao de riscos relativa a 65 anos; estagios mantidos na referencia I.
    grade=np.linspace(float(a.idade_anos.min()),float(a.idade_anos.max()),200)
    if not (grade.min() <= 65 <= grade.max()):
        raise ValueError('Idade de referencia 65 fora da faixa observada.')
    b_grade=build_design_matrices([design_info],{'x':grade})[0]
    b_ref=build_design_matrices([design_info],{'x':[65.]})[0]
    coef=modelo_spline.params_.loc[nomes].to_numpy(float)
    cov=modelo_spline.variance_matrix_.loc[nomes,nomes].to_numpy(float)
    delta=np.asarray(b_grade)-np.asarray(b_ref)[0]
    loghr=delta@coef
    se=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',delta,cov,delta)))
    idade=a.idade_anos.to_numpy(float)
    fig,ax=plt.subplots(figsize=(11,6))
    ax.plot(grade,np.exp(loghr),color=VERDE,lw=2.8,label='Spline cúbica (df=4)')
    ax.fill_between(grade,np.exp(loghr-1.96*se),np.exp(loghr+1.96*se),color=VERDE,alpha=.14,label='IC 95%')
    b_lin=modelo_linear.params_['idade_10anos']
    ax.plot(grade,np.exp(b_lin*(grade-65)/10),color=VERMELHO,lw=2,ls='--',label='Modelo linear')
    ax.axhline(1,color=CINZA,ls=':',lw=1)
    ax.axvline(65,color=CINZA,ls=':',lw=1)
    ax.set(xlabel='Idade ao diagnóstico (anos)',ylabel='HR relativo à idade de 65 anos',
           title='Forma funcional da idade | linear versus spline')
    ax.set_yscale('log');ax.grid(axis='y',color='#E5EBF0');ax.legend(frameon=False)
    ax.title.set_color(AZUL);ax.title.set_fontweight('bold')
    fig.text(.13,.015,f'Teste de razão de verossimilhanças (df={gl}): p={p:.3g}. ICs condicionais ao modelo; sem validação externa.',color=CINZA,fontsize=10)
    fig.tight_layout()
    salvar(fig,'06c_idade_splines.png')
    return comparacao

def influencia(modelo,a,cols):
    r=modelo.compute_residuals(a[['tempo_dias','evento']+cols],kind='delta_beta')
    # DFBETA aproximado via influencia de primeira ordem; nao e refit leave-one-out.
    esc=np.sqrt((r**2).sum(axis=1))
    tab=pd.DataFrame({'id_paciente':a.loc[r.index,COL['id']].to_numpy(),
                      'indice_linha':r.index.to_numpy(),'norma_delta_beta':esc.to_numpy(float),
                      'evento':a.loc[r.index,'evento'].to_numpy()})
    tab=tab.sort_values('norma_delta_beta',ascending=False).reset_index(drop=True)
    tab.to_csv(SAIDA/'06c_influencia_todos.csv',index=False,encoding='utf-8-sig')
    top=tab.head(min(15,len(tab))).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10.8,6.2))
    ax.barh(top.id_paciente,top.norma_delta_beta,color=[VERMELHO if i>=len(top)-3 else VERDE for i in range(len(top))])
    ax.set(xlabel='Norma euclidiana de delta-beta (aproximação)',
           title='Observações potencialmente influentes | triagem')
    ax.grid(axis='x',color='#E5EBF0',alpha=.7)
    ax.title.set_color(AZUL);ax.title.set_fontweight('bold')
    fig.text(.14,.015,'Ranking descritivo; não implica exclusão automática. Avaliar integridade e refits individuais.',fontsize=10,color=CINZA)
    fig.tight_layout();salvar(fig,'06c_observacoes_influentes.png')
    return tab

def main():
    original,a=preparar()
    cols=['idade_10anos','estagio_II','estagio_III_IV']
    modelo=ajustar(a,cols)
    modelo.summary.to_csv(SAIDA/'06c_cox_linear.csv',encoding='utf-8-sig')
    print(f'Coorte original={len(original)} | elegíveis={len(a)} | eventos={int(a.evento.sum())} | excluídos={len(original)-len(a)}')
    print('Estágios:',a.grupo_estagio.value_counts().to_dict())
    print('ATENÇÃO: eventos por parâmetro e tamanhos dos grupos devem ser considerados.')
    problemas=[]
    for titulo,funcao in [('Schoenfeld',lambda:painel_ph(modelo,a,cols)),
                          ('Splines',lambda:splines_idade(a,modelo,cols)),
                          ('Influência',lambda:influencia(modelo,a,cols))]:
        try:
            resultado=funcao()
            print(f'\n{titulo}: concluído')
            if titulo=='Schoenfeld':
                print(resultado[['variavel','transformacao','test_statistic','p']].to_string(index=False))
            elif titulo=='Splines':
                print(resultado.to_string(index=False))
            else:print(resultado.head(5).to_string(index=False))
        except Exception as e:
            problemas.append(f'{titulo}: {type(e).__name__}: {e}')
            print(f'ERRO em {titulo}: {e}')
    relatorio=[
        'SCRIPT 06C | TCGA-PAAD | RELATORIO METODOLOGICO',
        f'Coorte original: {len(original)} | Elegíveis: {len(a)} | Óbitos: {int(a.evento.sum())} | Excluídos: {len(original)-len(a)}',
        'Modelo: Cox sem penalização, empates Efron (lifelines), idade por 10 anos e estagios I/II/III-IV.',
        'PH: teste de residuos de Schoenfeld escalonados com transformacoes rank e km; testes individuais.',
        'IMPORTANTE: o script nao produz um teste GLOBAL formal de proporcionalidade; nao inferir validacao global.',
        'Splines: spline cubica df=4 versus idade linear; LRT exploratorio (3 graus de liberdade).',
        'Influencia: delta-beta aproximado; ranking nao autoriza exclusoes automaticas.',
        'Concordancia e AIC sao aparentes (mesma coorte); nao demonstram desempenho externo.',
        'Nao foram auditados origem do tempo, adequacao da censura, harmonizacao de estagios e vies de selecao.',
        'Resultados nao equivalem a validacao clinica, causal, preditiva externa ou ferramenta assistencial.',
        'Possivel instabilidade: grupo III-IV pequeno. Revisar codificacao Stage 0/0a e exclusoes.',
        'Erros em componentes: '+ ('; '.join(problemas) if problemas else 'nenhum')]
    (SAIDA/'06c_relatorio_metodologico.txt').write_text('\n'.join(relatorio),encoding='utf-8')
    print('\n'+'\n'.join(relatorio));print('\nResultados:',SAIDA)
    if problemas:raise RuntimeError('Algumas etapas falharam; consulte 06c_relatorio_metodologico.txt')

if __name__=='__main__':
    main()
