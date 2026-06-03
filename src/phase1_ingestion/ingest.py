# src/phase1_ingestion/ingest.py
import pandas as pd
import yaml
import os
from clean_text import clean_pbi_text

# --- LA LISTA DORADA (CURACIÓN DE DATOS) ---
# Solo permitimos proyectos que la auditoría demostró empíricamente que usan Fibonacci
GOLDEN_PROJECTS = [
    "278964.csv",
    "10171280.csv",
    "21149814.csv",
    "10171270.csv",
    "10171263.csv",
    "4456656.csv",
    "7128869.csv",
    "7071551.csv"
]

def main():
    print("🚀 Iniciando Fase 1: Ingesta Curada (Solo Proyectos Ágiles)...")
    
    # 1. Cargar configuración
    with open("config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    raw_folder = config["paths"]["raw_data_folder"]
    processed_path = config["paths"]["processed_data"]

    print(f"📦 Buscando proyectos en la Lista Dorada...")
    
    # 2. Leer y fusionar SOLO los CSVs de la Lista Dorada
    df_list = []
    for filename in GOLDEN_PROJECTS:
        full_path = os.path.join(raw_folder, filename)
        if os.path.exists(full_path):
            try:
                df_temp = pd.read_csv(full_path)
                # Guardamos el ID del proyecto por si queremos auditar después
                df_temp['project_id'] = filename.replace('.csv', '') 
                df_list.append(df_temp)
            except Exception as e:
                print(f"⚠️ Error leyendo {filename}: {e}")
        else:
            print(f"⚠️ Archivo no encontrado: {filename}")
            
    if not df_list:
        print("❌ Error: No se pudo cargar ningún proyecto de la Lista Dorada.")
        return

    df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"📊 Fusión completada. Registros curados totales: {len(df)}")

    # 3. Transformación de Columnas (Adaptación a nuestro Pipeline)
    print("⚙️ Adaptando esquema de columnas...")
    # Llenar vacíos para evitar errores al concatenar
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # Crear nuestra columna 'user_story' uniendo título y descripción
    df['user_story'] = df['title'] + " - " + df['description']
    
    # Renombrar 'storypoints' a 'target'
    df.rename(columns={'storypoints': 'target'}, inplace=True)

    # 4. Limpieza
    print("🧹 Limpiando texto y filtrando nulos...")
    df['user_story'] = df['user_story'].apply(clean_pbi_text)
    
    # Eliminar filas sin Story Points o con texto vacío
    filas_antes = len(df)
    df.dropna(subset=['user_story', 'target'], inplace=True)
    df.drop_duplicates(subset=['user_story'], keep='first', inplace=True)
    df = df[df['user_story'].str.strip() != "-"] # Filtrar si solo quedó el guión
    df = df[df['user_story'].str.strip() != ""]
    filas_despues = len(df)
    
    print(f"🗑️ Se eliminaron {filas_antes - filas_despues} registros vacíos o sin estimación.")

    # Nos quedamos solo con las columnas que le importan al resto del pipeline
    # IMPORTANTE: Ahora conservamos el 'project_id' para trazabilidad
    df_final = df[['project_id', 'user_story', 'target']].copy()

    # 5. Guardado en Parquet
    print(f"💾 Guardando dataset CURADO en {processed_path}...")
    # Crear carpeta processed si no existe
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_final.to_parquet(processed_path, index=False)
    
    print(f"✅ Fase 1 completada exitosamente. ¡{len(df_final)} PBIs de alta pureza listos para el Juez IA!")

if __name__ == "__main__":
    main()