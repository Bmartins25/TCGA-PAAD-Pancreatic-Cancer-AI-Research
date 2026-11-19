
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent

pasta_clinica = base / "clinical.project-tcga-paad.2026-11-18"
pasta_rnaseq = base / "RNAseq_STAR_Counts"

arquivos = [
    "clinical.tsv",
    "follow_up.tsv",
    "exposure.tsv",
    "family_history.tsv",
    "pathology_detail.tsv"
]

print("=== PILOTO TCGA-PAAD ===")

for nome in arquivos:
    caminho = pasta_clinica / nome

    if not caminho.exists():
        print(f"{nome}: NÃO ENCONTRADO")
        continue

    df = pd.read_csv(caminho, sep="\t", low_memory=False)

    print(f"\nArquivo: {nome}")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")
    print("Campos:", df.columns.tolist())

arquivos_rna = list(
    pasta_rnaseq.rglob(
        "*.rna_seq.augmented_star_gene_counts.tsv"
    )
)

print("\nArquivos RNA-seq:", len(arquivos_rna))
