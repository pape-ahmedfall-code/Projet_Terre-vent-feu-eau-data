import duckdb
import os
import time

DATA_DIR = r"data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

print("🦆 Initialisation du moteur DuckDB...")
start_time = time.time()

con = duckdb.connect(database=os.path.join(PROCESSED_DIR, "fire_data.duckdb"))

file_target = os.path.join(PROCESSED_DIR, "target_matrix_negative_sampling.parquet")
file_features = os.path.join(PROCESSED_DIR, "dataset_final_features_advanced.parquet")

# 1. Inspection automatique des colonnes disponibles dans le fichier de features
cols_features = [col[0] for col in con.execute(f"DESCRIBE SELECT * FROM '{file_features}'").fetchall()]
print(f"ℹ️ Colonnes détectées dans dataset_final_features_advanced : {cols_features[:5]}... (+{len(cols_features)-5} autres)")

# Sélection dynamique de quelques features météo réelles
meteo_cols = [c for c in cols_features if c not in ['code_insee', 'DATE']][:5]
meteo_select_sql = ", ".join([f"meteo.{c}" for c in meteo_cols])

# 2. Requête SQL d'ingestion et de jointure
query = f"""
CREATE OR REPLACE TABLE consolidated_fire_risk AS
SELECT 
    COALESCE(target.code_insee, meteo.code_insee) AS code_insee,
    CAST(COALESCE(target.DATE, meteo.DATE) AS DATE) AS date_evt,
    COALESCE(target.TARGET, 0) AS target_fire,
    EXTRACT(YEAR FROM CAST(COALESCE(target.DATE, meteo.DATE) AS DATE)) AS annee,
    EXTRACT(MONTH FROM CAST(COALESCE(target.DATE, meteo.DATE) AS DATE)) AS mois
    {', ' + meteo_select_sql if meteo_cols else ''}

FROM '{file_target}' AS target
FULL OUTER JOIN '{file_features}' AS meteo
    ON target.code_insee = meteo.code_insee 
   AND target.DATE = meteo.DATE;
"""

print("⚡ Exécution de la jointure SQL massive (INSEE x Date)...")
con.execute(query)

count_res = con.execute("SELECT COUNT(*) FROM consolidated_fire_risk").fetchone()[0]
sample_res = con.execute("SELECT * FROM consolidated_fire_risk LIMIT 5").fetchdf()

elapsed = time.time() - start_time

print(f"✅ Jointure SQL réussie en {elapsed:.2f} secondes !")
print(f"📊 Nombre total de lignes consolidées : {count_res:,}")
print("\n--- 🔍 Aperçu de la table SQL `consolidated_fire_risk` ---")
print(sample_res)

con.close()