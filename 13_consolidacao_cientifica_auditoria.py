#!/usr/bin/env python3
"""Etapa 13: auditoria documental e consolidação científica TCGA-PAAD.
Não treina modelos, não altera dados de origem e não infere validação externa.
"""
import argparse, csv, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path

ETAPAS = {
 '07':['07_resultados_por_dobra.csv','07_resumo_modelos.csv'],
 '08':['08_metricas_por_dobra.csv','08_resumo_metricas.csv'],
 '09':['09_pares_clinico_rnaseq.csv','09_mapeamento_gdc.csv'],
 '09b':['09b_matriz_contagens_brutas.csv','09b_pacientes_amostras_validos.csv'],
 '09c':['09c_qc_amostras.csv'],
 '09d':['09d_resumo.json'],
 '10':['10_resultados_por_dobra.csv'],
 '10b':['10b_resumo_diferencas.csv'],
 '11':['11_resumo.json','11_frequencia_selecao_genes.csv'],
 '12':['12_resumo.json','12_diferencas_pareadas.csv','12_resumo_subgrupos.csv']
}
# Nomes de etapas intermediárias podem variar conforme versões anteriores.
# Ausência de arquivo de nome esperado = não verificado, nunca prova de falha.

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def readcsv(path):
 with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def fmt(x):
 return 'não disponível' if x is None else f'{x:.4f}'.replace('.',',')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--diretorio',type=Path,default=Path(__file__).resolve().parent)
 ap.add_argument('--saida',type=Path,default=None)
 args=ap.parse_args(); raiz=args.diretorio.resolve(); saida=(args.saida or raiz/'resultados_consolidacao_13').resolve(); saida.mkdir(parents=True,exist_ok=True)
 fontes=[]
 for p in sorted(raiz.glob('*')):
  if p.is_file() and (p.name[:2].isdigit()) and not p.name.startswith('13_'):
   fontes.append({'arquivo':p.name,'tamanho_bytes':p.stat().st_size,'sha256':sha256(p)})
 with (saida/'13_manifesto_sha256.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['arquivo','tamanho_bytes','sha256']);w.writeheader();w.writerows(fontes)
 inventario=[]
 for etapa, nomes in ETAPAS.items():
  for nome in nomes:
   existe=(raiz/nome).is_file()
   inventario.append({'etapa':etapa,'arquivo_esperado':nome,'status':'localizado' if existe else 'não localizado pelo nome esperado'})
 with (saida/'13_inventario_etapas.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['etapa','arquivo_esperado','status']);w.writeheader();w.writerows(inventario)
 alertas=[]; verificacoes=[]
 resumo12=raiz/'12_resumo.json'
 if not resumo12.exists(): raise SystemExit('ERRO: 12_resumo.json ausente. Informe --diretorio com os resultados originais.')
 r=json.loads(resumo12.read_text(encoding='utf-8-sig'))
 def check(nome, ok, detalhe):
  verificacoes.append({'verificacao':nome,'resultado':'OK' if ok else 'ATENÇÃO','detalhe':detalhe})
  if not ok: alertas.append(nome+': '+detalhe)
 check('Etapa 12 concluída',str(r.get('status','')).startswith('concluida'),str(r.get('status')))
 check('Três modelos documentados',set(r.get('modelos',[]))=={'clinico','integrado','transcriptomico'},str(r.get('modelos')))
 check('172 pacientes distintos',r.get('pacientes_distintos')==172,str(r.get('pacientes_distintos')))
 check('6 dobras registradas',r.get('dobras')==6,str(r.get('dobras')))
 comp={x['comparacao']:x for x in r.get('comparacoes',[])}
 check('Três comparações pareadas',len(comp)==3,str(list(comp)))
 for nome,item in comp.items():
  check('Dobras comparadas: '+nome,item.get('n_dobras')==6,str(item.get('n_dobras')))
  check('Sinais das diferenças: '+nome,item.get('dobras_positivas',0)+item.get('dobras_negativas',0)==6,str((item.get('dobras_positivas'),item.get('dobras_negativas'))))
 csvcomp=raiz/'12_diferencas_pareadas.csv'
 if csvcomp.exists():
  linhas=readcsv(csvcomp)
  check('18 linhas de diferenças por dobra',len(linhas)==18,str(len(linhas)))
 else: check('CSV de diferenças por dobra',False,'arquivo ausente')
 modelos={}
 sub=raiz/'12_resumo_subgrupos.csv'
 if sub.exists():
  for linha in readcsv(sub):
   if linha.get('estrato')=='geral' and linha.get('grupo')=='todos':
    try: modelos[linha['modelo']]=float(linha['c_index_medio'])
    except (KeyError,ValueError): pass
 check('C-index dos três modelos recuperado',len(modelos)==3,str(modelos))
 if len(modelos)==3:
  check('Diferenças concordam com C-index (tolerância 0,0001)',all(abs((modelos[a]-modelos[b])-float(item['delta_medio']))<0.0001 for nome,item in comp.items() for a,b in [nome.split(' - ')]), 'comparação entre resumos das etapas 12')
 report={'etapa':'13','status':'auditoria documental executada','data_utc':datetime.now(timezone.utc).isoformat(),
 'origem':str(raiz),'arquivos_inventariados':len(fontes),'pacientes_distintos_etapa12':r.get('pacientes_distintos'),
 'dobras_etapa12':r.get('dobras'),'cindex_medio':modelos,'comparacoes_etapa12':r.get('comparacoes'),
 'verificacoes':verificacoes,'alertas':alertas,'limites':['Etapa 13 não reexecuta modelos nem audita o código-fonte de cada etapa.',
 'Os arquivos 09b/09c/09d/10/10b podem ter nomes diferentes dos esperados; ausência no inventário não comprova ausência de resultados.',
 'Dobras repetidas não são observações independentes; DP descritivo não é intervalo de confiança.',
 'Sem validação externa, inferência causal, benefício clínico ou superioridade estatística demonstrados.',
 'A ausência de controle de lote, temporalidade e pureza tumoral limita a interpretação biológica.',
 'Amostra e resultados são específicos da coorte disponível.']}
 (saida/'13_resumo.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 with (saida/'13_verificacoes.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['verificacao','resultado','detalhe']);w.writeheader();w.writerows(verificacoes)
 lines=['ETAPA 13 | CONSOLIDAÇÃO CIENTÍFICA E AUDITORIA DOCUMENTAL | TCGA-PAAD','',
 'OBJETIVO','Consolidar evidências existentes sem refazer ajuste, otimização ou validação externa.','',
 'COORTE E ESTRATÉGIA',f"Pacientes distintos nas predições da etapa 12: {r.get('pacientes_distintos')}.",f"Dobras de avaliação: {r.get('dobras')} (repetições dependentes).",'',
 'DISCRIMINAÇÃO (C-INDEX MÉDIO DESCRITIVO)']
 for m in ['clinico','integrado','transcriptomico']:lines.append(f'{m}: {fmt(modelos.get(m))}')
 lines+=['','COMPARAÇÕES PAREADAS (DESCRITIVAS)']
 for nome,item in comp.items():lines.append(f"{nome}: delta médio {fmt(item.get('delta_medio'))}; positivas {item.get('dobras_positivas')}/6; negativas {item.get('dobras_negativas')}/6.")
 lines+=['','INTERPRETAÇÃO','Na configuração avaliada, a integração transcriptômica não melhorou o C-index médio em relação ao modelo clínico.',
 'A diferença observada é descritiva, não evidência de inferioridade ou superioridade estatística.','',
 'LIMITAÇÕES E CONDUTAS']+['- '+s for s in report['limites']]
 lines+=['','PENDÊNCIAS DE AUDITORIA']+(['- '+s for s in alertas] if alertas else ['- Nenhuma inconsistência encontrada nas verificações automáticas implementadas.'])
 lines+=['','PRÓXIMOS PASSOS CIENTÍFICOS','Revisar temporalidade e definição do desfecho, qualidade biológica, efeitos de lote e possível viés de seleção.',
 'Realizar validação externa independente quando houver dados compatíveis.',
 'Para inferência de desempenho, definir protocolo estatístico adequado a dependência entre repetições e pacientes.',
 'Considerar calibração e utilidade clínica em validação prospectivamente especificada.']
 (saida/'13_relatorio_cientifico.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('\nETAPA 13 | CONSOLIDAÇÃO CIENTÍFICA')
 print('Arquivos inventariados:',len(fontes),'| verificações:',len(verificacoes),'| alertas:',len(alertas))
 for m,v in modelos.items():print(f'  {m}: C-index {v:.6f}')
 print('Resultados:',saida)
 print('IMPORTANTE: auditoria documental, não validação biológica, clínica ou externa.')

if __name__=='__main__':main()
