
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
pasta = base / "clinical.project-tcga-paad.2026-11-18"

df = pd.read_csv(
    pasta / "clinical.tsv",
    sep="\t",
    low_memory=False,
    na_values=["'--", "--", "Not Reported", "not reported"]
)

id_paciente = "cases.submitter_id"

print("=== CONSOLIDAÇÃO TCGA-PAAD ===")

print("Registros clínicos:", len(df))
print("Pacientes únicos:", df[id_paciente].nunique())
print("Registros duplicados por paciente:",
      df.duplicated(id_paciente).sum())

variaveis = [
    id_paciente,
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_pathologic_stage"
]

clinico = df[variaveis].copy()

for coluna in variaveis[2:5]:
    clinico[coluna] = pd.to_numeric(
        clinico[coluna], errors="coerce"
    )

# Consolidar informações demográficas por paciente.
# Para diagnóstico e seguimento, esta é apenas
# uma consolidação exploratória inicial.
def primeiro_valido(serie):
    valores = serie.dropna()
    return valores.iloc[0] if not valores.empty else pd.NA

pacientes = clinico.groupby(
    id_paciente, as_index=False
).agg({
    coluna: primeiro_valido
    for coluna in variaveis[1:]
})

print("\nPacientes consolidados:", len(pacientes))
print("\nSituação vital:")
print(pacientes["demographic.vital_status"].value_counts(
    dropna=False
))

print("\nDados ausentes:")
print(pacientes.isna().sum())

saida = base / "pacientes_tcga_paad_exploratorio.csv"
pacientes.to_csv(saida, index=False)

print("\nArquivo gerado:", saida)
