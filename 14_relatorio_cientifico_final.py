#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Etapa 14: relatório científico rastreável do piloto TCGA-PAAD.
Uso: python 14_relatorio_cientifico_final.py --diretorio "."
Não treina modelos, não baixa dados e não lê matrizes RNA-seq grandes.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

PASTAS = {
    '09': 'resultados_integracao_09',
    '09b': 'resultados_rnaseq_09b',
    '09c': 'resultados_rnaseq_09c',
    '09d': 'resultados_integracao_09d',
    '10': 'resultados_modelagem_10',
    '10b': 'resultados_modelagem_10b',
    '11': 'resultados_interpretabilidade_11',
    '12': 'resultados_robustez_12',
    '13': 'resultados_consolidacao_13',
}
# Apenas artefatos pequenos: jamais abrir matrizes ou arquivos brutos.
EXT = {'.json', '.csv', '.txt', '.md'}
LIMITE = 8 * 1024 * 1024


def pequeno(path):
    try:
        return path.is_file() and path.suffix.lower() in EXT and path.stat().st_size <= LIMITE
    except OSError:
        return False


def ler_json(path, alertas):
    try:
        with path.open('r', encoding='utf-8-sig') as f:
            return json.load(f)
    except (OSError, ValueError, UnicodeError) as e:
        alertas.append(f'Falha ao ler JSON {path}: {e}')
        return None


def achar_pasta(raiz, nome):
    candidatos = [raiz / nome, raiz / 'resultados_robustez_12' / nome]
    return next((p for p in candidatos if p.is_dir()), None)


def coletar(raiz, alertas):
    inventario = []
    pastas = {}
    for etapa, nome in PASTAS.items():
        pasta = achar_pasta(raiz, nome)
        pastas[etapa] = pasta
        if pasta is None:
            alertas.append(f'Pasta não localizada: {nome}')
            continue
        for p in sorted(pasta.iterdir()):
            if p.is_file():
                try:
                    tamanho = p.stat().st_size
                except OSError:
                    continue
                inventario.append({'etapa': etapa, 'arquivo': str(p.relative_to(raiz)),
                                   'bytes': tamanho, 'inspecionavel': pequeno(p)})
    return pastas, inventario


def buscar_json(pasta, candidatos, alertas):
    if pasta is None:
        return None, None
    for nome in candidatos:
        p = pasta / nome
        if pequeno(p):
            return ler_json(p, alertas), p
    return None, None


def procurar_valor(obj, chaves, profundidade=0):
    if profundidade > 5:
        return None
    if isinstance(obj, dict):
        for chave in chaves:
            if chave in obj and isinstance(obj[chave], (int, float, str)):
                return obj[chave]
        for valor in obj.values():
            r = procurar_valor(valor, chaves, profundidade + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for valor in obj[:100]:
            r = procurar_valor(valor, chaves, profundidade + 1)
            if r is not None:
                return r
    return None


def num(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def ler_csv_cabecalho(path, alertas):
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            return next(csv.reader(f), [])
    except (OSError, UnicodeError, csv.Error) as e:
        alertas.append(f'Falha no cabeçalho CSV {path}: {e}')
        return []


def markdown_tabela(linhas):
    if not linhas:
        return '_Sem dados verificáveis._\n'
    cab = list(linhas[0].keys())
    esc = lambda x: str(x if x is not None else 'Não disponível').replace('|', '\\|').replace('\n', ' ')
    return '| ' + ' | '.join(cab) + ' |\n| ' + ' | '.join(['---'] * len(cab)) + ' |\n' + ''.join('| ' + ' | '.join(esc(r.get(c)) for c in cab) + ' |\n' for r in linhas)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diretorio', default='.', help='Pasta principal do projeto')
    ap.add_argument('--saida', default='resultados_relatorio_14', help='Pasta de saída relativa à raiz')
    args = ap.parse_args()
    raiz = Path(args.diretorio).expanduser().resolve()
    if not raiz.is_dir():
        print(f'ERRO: diretório inexistente: {raiz}', file=sys.stderr)
        return 2
    saida = Path(args.saida)
    saida = (raiz / saida).resolve() if not saida.is_absolute() else saida.resolve()
    if saida == raiz:
        print('ERRO: saída não pode ser a pasta raiz.', file=sys.stderr)
        return 2
    saida.mkdir(parents=True, exist_ok=True)
    alertas = []
    pastas, inventario = coletar(raiz, alertas)
    # A etapa 13 pode ter sido criada dentro da etapa 12 ou na raiz.
    p13 = pastas['13']
    dados13, origem13 = buscar_json(p13, ['13_resumo.json', '13_consolidacao.json', '13_resumo_consolidacao.json'], alertas)
    dados12, origem12 = buscar_json(pastas['12'], ['12_resumo.json'], alertas)
    dados11, origem11 = buscar_json(pastas['11'], ['11_resumo.json'], alertas)
    dados10, origem10 = buscar_json(pastas['10'], ['10_resumo.json'], alertas)
    dados09d, origem09d = buscar_json(pastas['09d'], ['09d_resumo.json'], alertas)
    dados09b, origem09b = buscar_json(pastas['09b'], ['09b_resumo.json'], alertas)
    dados09c, origem09c = buscar_json(pastas['09c'], ['09c_resumo.json'], alertas)
    dados = {'09b': dados09b, '09c': dados09c, '09d': dados09d, '10': dados10,
             '11': dados11, '12': dados12, '13': dados13}
    origens = {'09b': origem09b, '09c': origem09c, '09d': origem09d, '10': origem10,
               '11': origem11, '12': origem12, '13': origem13}
    for etapa in ['09b', '09c', '09d', '10', '11', '12']:
        if dados[etapa] is None:
            alertas.append(f'Resumo JSON não identificado para etapa {etapa}; dados não serão presumidos.')
    if dados13 is None:
        alertas.append('Resumo JSON da etapa 13 não localizado por nome conhecido; confira o conteúdo da pasta.')
    # Métricas publicadas no terminal da etapa 13: referência apenas para conferência,
    # não são tratadas como evidência verificada caso os arquivos não as confirmem.
    referencia = {'clinico': 0.550866, 'integrado': 0.538341, 'transcriptomico': 0.486051}
    metricas = []
    for modelo, esperado in referencia.items():
        valor = None
        fonte = None
        # Procura apenas por chaves inequívocas, sem misturar resultados de outros modelos.
        for etapa in ['13', '12', '10']:
            obj = dados.get(etapa)
            if not isinstance(obj, dict):
                continue
            sec = obj.get(modelo)
            if isinstance(sec, dict):
                valor = num(procurar_valor(sec, ['c_index_medio', 'cindex_medio', 'c_index_mean']))
            if valor is None:
                for chave in [f'{modelo}_c_index', f'c_index_{modelo}', f'{modelo}_c_index_medio']:
                    if chave in obj:
                        valor = num(obj[chave])
                        break
            if valor is not None:
                fonte = str(origens[etapa].relative_to(raiz)) if origens[etapa] else None
                break
        if valor is None:
            alertas.append(f'C-index {modelo}: não extraído de resumo JSON conhecido; valor do terminal não foi adotado automaticamente.')
        elif abs(valor - esperado) > 0.0001:
            alertas.append(f'C-index {modelo}: JSON={valor:.6f} difere do terminal anterior={esperado:.6f}.')
        metricas.append({'modelo': modelo, 'c_index_medio': round(valor, 6) if valor is not None else None,
                         'fonte': fonte or 'Não verificada'})
    # Contagens só quando há uma chave identificável.
    contagens = []
    for etapa, chaves in [
        ('09d', ['pacientes_clinicos', 'pacientes_clinica']),
        ('09b', ['pacientes_com_matriz', 'pacientes_com_rna_tumor_primario']),
        ('09d', ['pacientes_pareados_desfecho_valido', 'pacientes_pareados']),
        ('11', ['pacientes_elegiveis'])
    ]:
        v = procurar_valor(dados.get(etapa), chaves) if dados.get(etapa) is not None else None
        contagens.append({'etapa': etapa, 'indicador': '/'.join(chaves), 'valor': v if v is not None else 'Não disponível'})
    # Verificação leve de CSVs: só cabeçalho e apenas até 8 MB.
    csvs = []
    for item in inventario:
        p = raiz / item['arquivo']
        if p.suffix.lower() == '.csv' and item['inspecionavel']:
            csvs.append({'arquivo': item['arquivo'], 'colunas': ', '.join(ler_csv_cabecalho(p, alertas)[:25])})
    resumo = {
        'etapa': '14', 'data_geracao': datetime.now().astimezone().isoformat(),
        'projeto': 'TCGA-PAAD', 'natureza': 'relatorio documental exploratorio',
        'diretorio': str(raiz), 'fontes_json': {k: str(v.relative_to(raiz)) if v else None for k, v in origens.items()},
        'metricas_verificadas': metricas, 'contagens_verificadas': contagens,
        'arquivos_inventariados': len(inventario), 'alertas': alertas,
        'validacao_externa': False, 'auditoria_sha256_global': False
    }
    (saida / '14_resumo.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    with (saida / '14_inventario.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['etapa', 'arquivo', 'bytes', 'inspecionavel'])
        w.writeheader(); w.writerows(inventario)
    with (saida / '14_metricas.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['modelo', 'c_index_medio', 'fonte'])
        w.writeheader(); w.writerows(metricas)
    with (saida / '14_cabecalhos_csv.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['arquivo', 'colunas'])
        w.writeheader(); w.writerows(csvs)
    sec = [
        '# TCGA-PAAD — Relatório científico consolidado (Etapa 14)',
        f'**Gerado em:** {resumo["data_geracao"]}',
        '**Escopo:** auditoria documental e síntese exploratória de dados clínicos e RNA-seq.',
        '## 1. Objetivo',
        'Consolidar resultados das etapas 09 a 13 com rastreabilidade documental, sem reprocessar RNA-seq nem retreinar modelos.',
        '## 2. Coorte e integração', markdown_tabela(contagens),
        'As contagens acima são exibidas somente quando identificadas nos resumos disponíveis; devem ser conferidas com as definições de coorte e endpoint.',
        '## 3. Desempenho preditivo exploratório', markdown_tabela(metricas),
        'C-index próximo de 0,5 indica discriminação limitada. Diferenças descritivas entre modelos não demonstram superioridade estatística.',
        'Os valores são omitidos quando não puderam ser extraídos de uma fonte estruturada conhecida.',
        '## 4. Interpretabilidade',
        'A etapa 11 avaliou contribuições ao log-risco de um modelo de Cox; tais contribuições não são efeitos causais nem validação de biomarcadores.',
        '## 5. Robustez',
        'As comparações entre dobras repetidas são dependentes. Não interpretar desvios-padrão entre dobras como intervalos de confiança independentes.',
        'Análises por subgrupos com poucos eventos ou dobras não avaliáveis não permitem conclusões estáveis.',
        '## 6. Limitações',
        '- Coorte retrospectiva observacional e amostra restrita ao TCGA-PAAD.',
        '- Necessário verificar temporalidade, censura, origem dos dados, efeitos de lote e qualidade biológica.',
        '- A etapa 09C foi exploratória; filtros, escalonamento e seleção para predição devem ocorrer exclusivamente dentro de cada dobra de treino.',
        '- Não foram demonstradas validação externa, utilidade clínica, causalidade ou generalização.',
        '- A auditoria SHA-256 global 13B foi interrompida e não é considerada concluída.',
        '## 7. Rastreabilidade e pendências',
        f'Foram inventariados {len(inventario)} arquivos diretos nas pastas encontradas, sem leitura de matrizes brutas.',
        '### Alertas e verificações pendentes',
        '\n'.join('- ' + a for a in alertas) if alertas else 'Nenhum alerta nas verificações documentais executadas.',
        '## 8. Conclusão',
        'Este documento consolida evidências exploratórias disponíveis. Não constitui validação clínica, biológica ou externa dos modelos.',
        'Recomenda-se revisar as fontes, completar a avaliação de calibração e validação externa quando houver coorte independente apropriada.',
    ]
    (saida / '14_relatorio_cientifico.md').write_text('\n\n'.join(sec) + '\n', encoding='utf-8')
    print('ETAPA 14 | CONSOLIDAÇÃO CIENTÍFICA DOCUMENTAL')
    print('Arquivos inventariados:', len(inventario))
    print('Alertas/pendências:', len(alertas))
    print('Resultados:', saida)
    for m in metricas:
        print(f"  {m['modelo']}: {m['c_index_medio'] if m['c_index_medio'] is not None else 'não verificado'}")
    print('NOTA: a auditoria global 13B permanece não concluída; este script não a substitui.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
