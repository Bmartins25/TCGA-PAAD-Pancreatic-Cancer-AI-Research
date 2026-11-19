#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etapa 13B: auditoria documental global, sem reexecutar modelos."""
import argparse, csv, hashlib, json, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STAGES = ('08','09','09b','09c','09d','10','10b','11','12','13')
STAGE_RE = re.compile(r'^(08|09|10|11|12|13)([bcd])?(?:[_\-.]|$)', re.I)
SKIP_DIRS = {'.git','.venv','venv','__pycache__','node_modules','resultados_auditoria_13b'}
EXTS = {'.csv','.json','.txt','.png','.jpg','.jpeg','.pdf','.py','.parquet','.tsv','.md','.xlsx','.pkl','.joblib','.html'}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),''): h.update(chunk)
    return h.hexdigest()

def stage_for(path):
    for part in reversed(path.parts):
        p=part.lower()
        m=STAGE_RE.match(p)
        if m:
            stage=m.group(1)+ (m.group(2) or '')
            if stage in STAGES: return stage
        # folder names such as resultados_robustez_12 / resultados_rnaseq_09b
        m=re.search(r'(?:^|[_\-.])(08|09|10|11|12|13)([bcd])?(?:$|[_\-.])',p)
        if m:
            stage=m.group(1)+(m.group(2) or '')
            if stage in STAGES:return stage
    return ''

def csv_info(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        sample=f.read(8192); f.seek(0)
        try: dialect=csv.Sniffer().sniff(sample,delimiters=',;\t')
        except csv.Error: dialect=csv.excel
        reader=csv.reader(f,dialect)
        header=next(reader,[])
        n=sum(1 for _ in reader)
    return {'linhas_dados':n,'colunas':len(header),'cabecalho':header[:30]}

def write_csv(path, rows, fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Auditoria global das etapas 08 a 13 do piloto TCGA-PAAD')
    ap.add_argument('--raiz',type=Path,default=Path(__file__).resolve().parent,help='Pasta principal do piloto')
    ap.add_argument('--saida',type=Path,default=None,help='Pasta de saída (padrão: <raiz>/resultados_auditoria_13b)')
    ap.add_argument('--sem-hash',action='store_true',help='Evita SHA-256 em arquivos grandes; registra como NAO_CALCULADO')
    args=ap.parse_args(); root=args.raiz.expanduser().resolve()
    if not root.is_dir(): ap.error('Diretório --raiz não existe: '+str(root))
    out=(args.saida or root/'resultados_auditoria_13b').expanduser().resolve()
    if out==root: ap.error('A pasta de saída não pode ser a raiz do projeto')
    out.mkdir(parents=True,exist_ok=True)
    inventory=[]; checks=[]; errors=[]; bystage=Counter(); json_data={}
    def check(name,status,detail): checks.append({'verificacao':name,'status':status,'detalhes':detail})
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.relative_to(root).parts):continue
        if out==path or out in path.parents:continue
        if path.suffix.lower() not in EXTS:continue
        rel=path.relative_to(root); stage=stage_for(rel)
        info={'etapa':stage or 'nao_classificado','arquivo':str(rel),'extensao':path.suffix.lower(), 'bytes':path.stat().st_size,'sha256':'NAO_CALCULADO' if args.sem_hash else '', 'validacao':'nao_aplicavel','detalhe':''}
        try:
            if not args.sem_hash:info['sha256']=sha256(path)
            if path.suffix.lower()=='.json':
                with path.open('r',encoding='utf-8-sig') as f: obj=json.load(f)
                info['validacao']='json_valido';info['detalhe']='tipo='+type(obj).__name__
                if path.name.lower().endswith('resumo.json'):json_data.setdefault(stage,[]).append((str(rel),obj))
            elif path.suffix.lower() in ('.csv','.tsv'):
                d=csv_info(path);info['validacao']='csv_lido';info['detalhe']=f"linhas={d['linhas_dados']}; colunas={d['colunas']}"
            elif path.stat().st_size==0:info['validacao']='vazio'
        except Exception as e:
            info['validacao']='erro';info['detalhe']=f'{type(e).__name__}: {e}';errors.append(str(rel)+': '+info['detalhe'])
        inventory.append(info)
        if stage:bystage[stage]+=1
    for stage in STAGES:
        count=bystage[stage]
        check('cobertura_etapa_'+stage,'OK' if count else 'PENDENTE',f'{count} arquivo(s) identificados; ausência não implica que a etapa não foi executada')
    check('arquivos_legiveis','FALHA' if errors else 'OK',f'{len(errors)} arquivo(s) com erro de leitura/estrutura')
    check('hashes','NAO_EXECUTADO' if args.sem_hash else 'OK','SHA-256 de todos os arquivos inventariados' if not args.sem_hash else 'Use sem --sem-hash para verificar integridade futura')
    summaries=json_data.get('12',[])
    if not summaries:
        check('resumo_etapa_12','PENDENTE','12_resumo.json não localizado na raiz pesquisada')
    else:
        matches=[(p,o) for p,o in summaries if Path(p).name.lower()=='12_resumo.json']
        if not matches:check('resumo_etapa_12','PENDENTE','JSON(s) da etapa 12 encontrados, mas nenhum 12_resumo.json')
        else:
            p,o=matches[0]; required=['pacientes_distintos','dobras','comparacoes']
            missing=[k for k in required if not isinstance(o,dict) or k not in o]
            check('resumo_etapa_12','PENDENTE' if missing else 'OK',f'{p}; campos ausentes: {missing}')
    # Verificações entre etapas são documentais, não equivalem a validação de métricas/modelos.
    check('validacao_externa','NAO_AVALIADO','Auditoria documental não confirma validação externa, calibração ou utilidade clínica')
    check('vazamento_de_dados','NAO_AVALIADO','Não é possível excluir vazamento apenas com nomes/formatos dos artefatos')
    write_csv(out/'13b_inventario_sha256.csv',inventory,['etapa','arquivo','extensao','bytes','sha256','validacao','detalhe'])
    write_csv(out/'13b_verificacoes.csv',checks,['verificacao','status','detalhes'])
    summary={'etapa':'13B','tipo':'auditoria documental e integridade','data_utc':datetime.now(timezone.utc).isoformat(),'raiz':str(root),'arquivos_inventariados':len(inventory),'arquivos_por_etapa':dict(bystage),'verificacoes':dict(Counter(c['status'] for c in checks)),'erros':errors,'observacoes':['Não reexecuta modelos nem confirma reprodutibilidade computacional.','Etapas ausentes exigem inspeção dos caminhos e nomes, não devem ser consideradas concluídas.','Hash SHA-256 permite comparação futura quando houver um manifesto de referência.','Verificações automáticas de leitura e cobertura não comprovam qualidade biológica ou validade clínica.']}
    (out/'13b_resumo.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['ETAPA 13B | AUDITORIA GLOBAL DOCUMENTAL | TCGA-PAAD','',f'Raiz: {root}',f'Arquivos inventariados: {len(inventory)}','', 'COBERTURA POR ETAPA']
    lines.extend(f'Etapa {s}: {bystage[s]} arquivo(s)' for s in STAGES)
    lines+=['','VERIFICAÇÕES']
    lines.extend(f"[{c['status']}] {c['verificacao']}: {c['detalhes']}" for c in checks)
    lines+=['','LIMITAÇÕES','Auditoria documental: não substitui reexecução, validação externa, auditoria de vazamento, controle de lote ou avaliação clínica.']
    (out/'13b_relatorio_auditoria.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ETAPA 13B | AUDITORIA GLOBAL DOCUMENTAL')
    print('Arquivos inventariados:',len(inventory))
    for s in STAGES:print(f'  {s}: {bystage[s]}')
    print('Verificações:',summary['verificacoes'])
    print('Resultados:',out)
    if errors:print('ATENÇÃO: arquivos com erro de leitura; consultar 13b_verificacoes.csv')
    if any(bystage[s]==0 for s in STAGES):print('ATENÇÃO: há etapas sem artefatos identificados; revisar caminhos/nomes.')
    return 1 if errors else 0
if __name__=='__main__':sys.exit(main())
