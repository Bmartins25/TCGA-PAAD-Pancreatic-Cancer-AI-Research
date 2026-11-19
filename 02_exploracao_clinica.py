
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
pasta = base / "clinical.project-tcga-paad.2026-11-18"

for nome in ["clinical.tsv", "follow_up.tsv"]:
    caminho = pasta / nome
    df = pd.read_csv(caminho, sep="\t", low_memory=False)

    print(f"\n=== {nome} ===")
    print("Linhas:", len(df))
    print("Colunas:", len(df.columns))

    termos = [
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
        "days_to_last_known_alive",
        "age_at_diagnosis",
        "ajcc_pathologic_stage",
        "submitter_id"
    ]

    colunas = [
        c for c in df.columns
        if any(termo in c.lower() for termo in termos)
    ]

    print("\nVariáveis encontradas:")
    for coluna in colunas:
        print("-", coluna)

    print("\nPrimeiros registros:")
    print(df[colunas].head(5).to_string(index=False))
