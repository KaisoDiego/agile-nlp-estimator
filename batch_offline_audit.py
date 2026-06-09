# batch_offline_audit.py

import os
import sys
import time
import yaml
import torch
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

# Interfaz de Consola Profesional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Asegurar que Python encuentre las rutas internas de tu proyecto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.phase2_router.prompts import SYSTEM_PROMPT_BATCH_JUDGE

console = Console()

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("config/settings.yaml", "r") as file:
    config = yaml.safe_load(file)

# --- 2. ESTRUCTURA Y FUNCIONES AUXILIARES ---
class QualityEvaluation(BaseModel):
    pbi_type: str = Field(..., description="Clasificación: 'User Story', 'Technical Task', 'Bug', 'Spike'")
    hu_q_score: float = Field(..., description="Nota de 1.0 a 5.0")
    defect_reasoning: str = Field(..., description="Justificación técnica de la nota")

def mean_pooling(model_output, attention_mask):
    """Comprime la salida neuronal de SBERT en un solo vector."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def snap_to_fibonacci_conservative(val):
    """Ajusta la predicción continua a la carta de Fibonacci más cercana."""
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    return min(fibs, key=lambda x: abs(x - val))

def main():
    console.print(Panel.fit("[bold cyan]🕵️‍♂️ Auditoría Forense MLOps (Modo Batch Offline)[/bold cyan]", border_style="cyan"))

    # Rutas de Archivos (Ajusta la ruta de entrada según tu CSV real)
    INPUT_CSV = "test/moodtab_backlog3_en.xlsx"  # O "data/moodtab_history_clean.csv"
    OUTPUT_CSV = "reports/auditoria_forense_moodtab3.csv"

    if not os.path.exists(INPUT_CSV):
        console.print(f"[bold red]❌ No se encontró el archivo de entrada: {INPUT_CSV}[/bold red]")
        return

    # --- 3. CARGA DE MODELOS (SBERT + LightGBM) ---
    console.print("[dim]📥 Cargando modelos en memoria (SBERT y LightGBM)...[/dim]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SBERT
    tokenizer = AutoTokenizer.from_pretrained(config["vectorizer"]["model_name"])
    sbert_model = AutoModel.from_pretrained(config["vectorizer"]["model_name"]).to(device)
    sbert_model.eval()

    # LightGBM
    artefactos = joblib.load(config["paths"]["model_save_path"])
    lgbm_model = artefactos["model"]
    label_encoder = artefactos["label_encoder"]
    
    # --- 4. CARGA DE DATOS (CRUCE RELACIONAL) ---
    console.print("[dim]📄 Leyendo y cruzando hojas del archivo Excel...[/dim]")
    
    # 1. Leemos solo las dos hojas que nos importan
    df_backlog = pd.read_excel(INPUT_CSV, sheet_name="Sprint 3 Backlog")
    df_detalles = pd.read_excel(INPUT_CSV, sheet_name="Listado de HUs")
    
    # 2. Nos quedamos solo con las columnas de texto de la segunda hoja para no duplicar datos
    df_detalles = df_detalles[["Título", "Historia de Usuario", "Criterios de Aceptación"]]
    
    # 3. Hacemos un JOIN (Cruce) uniendo ambas tablas por la columna "Título"
    df = pd.merge(df_backlog, df_detalles, on="Título", how="left")
    
    # 4. Limpiamos los nulos SOLO en las columnas de texto para que Pandas no explote
    for col in ["Título", "Historia de Usuario", "Criterios de Aceptación"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
            
    resultados = []

    # --- 5. BUCLE DE PROCESAMIENTO BATCH ---
    console.print(f"[bold green]🚀 Iniciando análisis de {len(df)} tickets históricos...[/bold green]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        
        task_audit = progress.add_task("[cyan]Analizando Backlog...", total=len(df))

        for index, row in df.iterrows():
            try:
                # 5.1. Extraer datos (Ahora cruzados con los criterios)
                ticket_id = str(row.get("ID HUs", f"PBI-{index}"))
                titulo = str(row.get("Título", ""))
                
                # Construir una descripción robusta extrayendo las nuevas columnas
                historia = str(row.get("Historia de Usuario", ""))
                criterios = str(row.get("Criterios de Aceptación", ""))
                descripcion = f"Historia: {historia}\nCriterios de Aceptación: {criterios}"
                
                # Extraer el esfuerzo (columna 'SP')
                esfuerzo_real = row.get("SP", np.nan)
                if pd.isna(esfuerzo_real) or esfuerzo_real == "":
                    esfuerzo_real = None
                
                texto_completo = f"{titulo}\n{descripcion}"
                if len(texto_completo.strip()) < 5:
                    progress.advance(task_audit)
                    continue

                # 5.2. Evaluación LLM Zero-Trust
                response = client.beta.chat.completions.parse(
                    model=config["llm_router"]["model_name"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_BATCH_JUDGE},
                        {"role": "user", "content": f"Evalúa este PBI:\n{texto_completo}"}
                    ],
                    temperature=0.0,
                    response_format=QualityEvaluation,
                )
                evaluacion = response.choices[0].message.parsed

                # 5.3. Vectorización SBERT
                encoded_input = tokenizer([texto_completo], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
                with torch.no_grad():
                    model_output = sbert_model(**encoded_input)
                embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()

                # 5.4. Preparar variables para LightGBM
                # Manejar tipos de PBI desconocidos con un fallback seguro
                try:
                    pbi_encoded = label_encoder.transform([evaluacion.pbi_type])[0]
                except ValueError:
                    pbi_encoded = label_encoder.transform(['Technical Task'])[0] # Fallback
                
                scores = np.array([[evaluacion.hu_q_score, pbi_encoded]])
                X_input = np.hstack((embedding, scores))

                # 5.5. Predicción Algorítmica
                pred_continua = max(1, lgbm_model.predict(X_input)[0])
                pred_fibonacci = snap_to_fibonacci_conservative(pred_continua)

                # 5.6. Guardar el registro
                resultados.append({
                    "Ticket_ID": ticket_id,
                    "Titulo": titulo[:50] + "...",
                    "Texto_Completo": texto_completo,  # <--- AGREGA ESTA LÍNEA
                    "Q_Score_DoR": evaluacion.hu_q_score,
                    "Tipo_Detectado": evaluacion.pbi_type,
                    "SP_Empirico_Historico": esfuerzo_real,
                    "SP_Inferencia_IA": pred_fibonacci,
                    "Justificacion_IA": evaluacion.defect_reasoning
                })

                # Sleep para no golpear los límites de OpenAI (Rate Limits)
                time.sleep(0.5) 
                progress.advance(task_audit)

            except Exception as e:
                # Si falla un ticket, lo ignoramos y seguimos
                console.log(f"[red]Error procesando ticket en fila {index}: {e}[/red]")
                progress.advance(task_audit)

    # --- 6. EXPORTACIÓN DE RESULTADOS ---
    os.makedirs("reports", exist_ok=True)
    df_resultados = pd.DataFrame(resultados)
    
    # 🛡️ MLOps Defense: Conversión segura de números para calcular la desviación
    if not df_resultados.empty and "SP_Empirico_Historico" in df_resultados.columns:
        # Convertir a numérico, los vacíos o textos raros se vuelven NaN
        df_resultados["SP_Empirico_Historico"] = pd.to_numeric(df_resultados["SP_Empirico_Historico"], errors='coerce')
        df_resultados["SP_Inferencia_IA"] = pd.to_numeric(df_resultados["SP_Inferencia_IA"], errors='coerce')
        
        # Calcular desviación: (Lo que dijo el equipo) - (Lo que dijo la IA)
        df_resultados["Desviacion_SP"] = df_resultados["SP_Empirico_Historico"] - df_resultados["SP_Inferencia_IA"]

    df_resultados.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    console.print("\n" + "="*60)
    console.print(f"✅ [bold green]Auditoría Finalizada Exitosamente[/bold green]")
    console.print(f"📊 Se procesaron [bold cyan]{len(resultados)}[/bold cyan] requerimientos válidos.")
    console.print(f"💾 Resultados exportados en: [bold yellow]{OUTPUT_CSV}[/bold yellow]")
    console.print("="*60 + "\n")

if __name__ == "__main__":
    main()