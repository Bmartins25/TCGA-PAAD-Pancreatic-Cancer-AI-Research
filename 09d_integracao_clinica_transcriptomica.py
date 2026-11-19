#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09D — integração auditável TCGA-PAAD. Python >=3.9; pandas, numpy.
Não treina modelos nem reutiliza transformações exploratórias da 09C como entrada preditiva.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

def ler_csv(caminho, **kwargs):
    if not caminho.is_file():
        raise FileNotFoundError(f'Arquivo obrigatório não encontrado: {caminho}')
    return pd.read_csv(caminho, encoding='utf-8-sig', **kwargs)

def normalizar_id(s):
    return s.astype('string').str.strip().str.upper()

def salvar(df, destino):
    df.to_csv(destino, index=False, encoding='utf-8-sig')
    print(f'Salvo: {destino.name} ({len(df)} linhas)')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clinico', type=Path, default=BASE/'pacientes_tcga_paad_exploratorio.csv')
    p.add_argument('--mapa', type=Path, default=BASE/'resultados_rnaseq_09b'/'09b_pacientes_amostras_validos.csv')
    p.add_argument('--contagens', type=Path, default=BASE/'resultados_rnaseq_09b'/'09b_matriz_contagens_brutas.csv.gz')
    p.add_argument('--qc', type=Path, default=BASE/'resultados_rnaseq_09c'/'09c_qc_amostras.csv')
    p.add_argument('--saida', type=Path, default=BASE/'resultados_integracao_09d')
    p.add_argument('--exportar-matriz', action='store_true', help='Exportar matriz bruta restrita a pacientes pareados (genes x pacientes).')
    args = p.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    cli = ler_csv(args.clinico)
    mapa = ler_csv(args.mapa, dtype=str)
    col_id = 'cases.submitter_id'
    if col_id not in cli.columns:
        raise ValueError(f'Clínico sem coluna {col_id}. Colunas: {list(cli.columns)}')
    if 'id_paciente' not in mapa.columns:
        raise ValueError(f'Mapa sem id_paciente. Colunas: {list(mapa.columns)}')
    cli['id_paciente'] = normalizar_id(cli[col_id])
    mapa['id_paciente'] = normalizar_id(mapa['id_paciente'])
    if cli.id_paciente.isna().any() or cli.id_paciente.duplicated().any():
        raise ValueError('IDs clínicos ausentes ou duplicados: revisar a consolidação antes da integração.')
    if mapa.id_paciente.isna().any() or mapa.id_paciente.duplicated().any():
        raise ValueError('IDs RNA-seq ausentes ou duplicados: resolver amostras múltiplas antes da integração.')

    print('09D | Lendo contagens brutas (sem normalização preditiva)...', flush=True)
    cont = ler_csv(args.contagens, index_col=0)
    cont.columns = [str(c).strip().upper() for c in cont.columns]
    if cont.columns.duplicated().any():
        raise ValueError('Colunas duplicadas na matriz de contagens.')
    if cont.index.duplicated().any():
        raise ValueError('Genes duplicados na matriz de contagens; não agregar automaticamente.')
    ids_matriz = pd.Index(cont.columns, name='id_paciente')
    ids_mapa = pd.Index(mapa.id_paciente, name='id_paciente')
    ids_clinicos = pd.Index(cli.id_paciente, name='id_paciente')
    if not ids_matriz.isin(ids_mapa).all():
        raise ValueError('Existem IDs da matriz não encontrados no mapeamento 09B; verificar antes de prosseguir.')
    ids_pareados = sorted(set(ids_matriz).intersection(ids_mapa).intersection(ids_clinicos))
    if not ids_pareados:
        raise ValueError('Nenhum paciente pareado. Conferir orientação da matriz e IDs.')

    # Auditoria por paciente: não presume que metadados equivalem a expressão válida.
    auditoria = cli[['id_paciente']].copy()
    auditoria['em_mapa_09b'] = auditoria.id_paciente.isin(ids_mapa)
    auditoria['em_matriz_bruta_09b'] = auditoria.id_paciente.isin(ids_matriz)
    auditoria['pareado_clinico_rnaseq'] = auditoria.id_paciente.isin(ids_pareados)
    auditoria['motivo'] = np.select(
        [~auditoria.em_mapa_09b, ~auditoria.em_matriz_bruta_09b],
        ['sem_vinculo_no_mapa_09b', 'sem_coluna_na_matriz_09b'], default='pareado')
    salvar(auditoria, args.saida/'09d_auditoria_pareamento.csv')

    combinado = cli.merge(mapa, on='id_paciente', how='left', validate='one_to_one', suffixes=('', '_rna'))
    combinado = combinado.merge(auditoria, on='id_paciente', how='left', validate='one_to_one')
    salvar(combinado, args.saida/'09d_coorte_clinica_auditada.csv')
    pareado = combinado[combinado.pareado_clinico_rnaseq].copy()

    # Desfecho: codificar apenas quando estado vital e tempo são válidos; nunca imputar.
    status_col = 'demographic.vital_status'
    morte_col = 'demographic.days_to_death'
    seguimento_col = 'diagnoses.days_to_last_follow_up'
    for c in (status_col, morte_col, seguimento_col):
        if c not in pareado.columns:
            raise ValueError(f'Campo clínico necessário ausente: {c}')
    status = pareado[status_col].astype('string').str.strip().str.lower()
    pareado['evento_obito'] = status.map({'dead': 1, 'alive': 0}).astype('Int64')
    morte = pd.to_numeric(pareado[morte_col], errors='coerce')
    seguimento = pd.to_numeric(pareado[seguimento_col], errors='coerce')
    pareado['tempo_dias'] = np.where(pareado.evento_obito.eq(1).fillna(False), morte,
                                    np.where(pareado.evento_obito.eq(0).fillna(False), seguimento, np.nan))
    pareado['tempo_dias'] = pd.to_numeric(pareado.tempo_dias, errors='coerce')
    pareado['desfecho_valido'] = pareado.evento_obito.notna() & pareado.tempo_dias.notna() & (pareado.tempo_dias > 0)
    pareado['motivo_desfecho'] = np.where(pareado.desfecho_valido, 'valido', 'evento_ou_tempo_ausente_invalido')
    salvar(pareado, args.saida/'09d_pacientes_pareados_clinicos.csv')

    # Verificação de integridade das contagens (sem filtragem global para modelagem).
    arr = cont.to_numpy(copy=False)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError('Matriz contém valores não numéricos.')
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError('Matriz contém valores negativos ou não finitos.')
    if not np.allclose(arr, np.rint(arr)):
        raise ValueError('Matriz não parece conter contagens inteiras brutas.')
    resumo_amostras = pd.DataFrame({
        'id_paciente': cont.columns,
        'contagens_totais': np.asarray(arr.sum(axis=0), dtype=np.float64),
        'genes_detectados': np.asarray((arr > 0).sum(axis=0), dtype=np.int64),
    })
    resumo_amostras['pareado_clinico'] = resumo_amostras.id_paciente.isin(ids_pareados)
    if args.qc.is_file():
        qc = ler_csv(args.qc)
        if 'id_paciente' in qc.columns:
            qc['id_paciente'] = normalizar_id(qc['id_paciente'])
            if not qc.id_paciente.duplicated().any():
                resumo_amostras = resumo_amostras.merge(qc, on='id_paciente', how='left', validate='one_to_one', suffixes=('_09d', '_09c'))
            else:
                print('AVISO: QC 09C tem IDs duplicados; não integrado automaticamente.')
        else:
            print('AVISO: QC 09C sem id_paciente; não integrado automaticamente.')
    else:
        print('AVISO: QC 09C não localizado; integração clínica-contagens prossegue com ressalva.')
    salvar(resumo_amostras, args.saida/'09d_auditoria_contagens_amostras.csv')

    campos = ['id_paciente', 'evento_obito', 'tempo_dias', 'desfecho_valido',
              'diagnoses.age_at_diagnosis', 'diagnoses.ajcc_pathologic_stage']
    campos = [c for c in campos if c in pareado.columns]
    salvar(pareado[campos], args.saida/'09d_base_clinica_para_modelagem_futura.csv')
    faltantes = pareado.isna().sum().rename('n_ausentes').reset_index().rename(columns={'index':'variavel'})
    faltantes['percentual'] = (100 * faltantes.n_ausentes / max(len(pareado), 1)).round(2)
    salvar(faltantes, args.saida/'09d_auditoria_ausencias.csv')
    if args.exportar_matriz:
        cont.loc[:, ids_pareados].rename_axis('gene_id').to_csv(
            args.saida/'09d_contagens_brutas_pareadas.csv.gz', compression='gzip')
        print('Matriz bruta pareada exportada, SEM filtragem ou normalização.')

    resumo = {
        'etapa':'09D', 'status':'integração auditada; não equivale a validação biológica ou preditiva',
        'pacientes_clinicos':int(len(cli)), 'pacientes_mapa_09b':int(len(mapa)),
        'pacientes_matriz_09b':int(len(ids_matriz)), 'pacientes_pareados':int(len(ids_pareados)),
        'pacientes_pareados_desfecho_valido':int(pareado.desfecho_valido.sum()),
        'genes_matriz_bruta':int(len(cont)), 'matriz_exportada':bool(args.exportar_matriz),
        'nota':'09C foi exploratória. Para validação preditiva, refazer filtragem/normalização/seleção somente no treino de cada dobra.',
        'fontes':{'clinico':str(args.clinico.resolve()),'mapa':str(args.mapa.resolve()),
                  'contagens':str(args.contagens.resolve()),'qc':str(args.qc.resolve()) if args.qc.is_file() else None}
    }
    (args.saida/'09d_resumo.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    rel = ('ETAPA 09D | TCGA-PAAD | AUDITORIA DE INTEGRAÇÃO\n\n' +
           '\n'.join(f'{k}: {v}' for k,v in resumo.items()) +
           '\n\nLIMITAÇÕES\n'
           'O pareamento é por identificador de paciente, não por data de coleta.\n'
           'Não há imputação, seleção automática de amostras, harmonização de lote ou modelagem.\n'
           'Tempo de sobrevida é derivado de dias até morte (óbito) ou último seguimento (vivo).\n'
           'Revisar origem temporal, definição do endpoint, qualidade de RNA e metadados técnicos.\n'
           'Não utilizar a matriz filtrada/logCPM/PCA da 09C como entrada direta de validação cruzada.\n')
    (args.saida/'09d_relatorio_metodologico.txt').write_text(rel, encoding='utf-8')
    print('\n'+rel+'\nResultados:', args.saida)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERRO 09D: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
