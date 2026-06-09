# app/api_webhook.py

import sys
import os
import yaml
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import csv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.phase2_router.prompts import SYSTEM_PROMPT_BATCH_JUDGE

# 1. Configuración Inicial
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("config/settings.yaml", "r") as file:
    config = yaml.safe_load(file)

app = FastAPI(
    title="Agile Tutor - Jira Webhook (Capa 1)",
    description="Microservicio que escucha eventos de Jira y audita la calidad de los PBIs en tiempo real."
)

# 2. Estructura de Salida del LLM (Igual que en tu juez)
class QualityEvaluation(BaseModel):
    pbi_type: str = Field(..., description="Clasificación: 'User Story', 'Technical Task', 'Bug', 'Spike'")
    hu_q_score: float = Field(..., description="Nota de 1.0 a 5.0")
    defect_reasoning: str = Field(..., description="Justificación técnica de la nota")

def auditar_calidad_pbi(texto: str):
    response = client.beta.chat.completions.parse(
        model=config["llm_router"]["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BATCH_JUDGE},
            {"role": "user", "content": f"Evalúa este PBI recién creado en Jira:\n{texto}"}
        ],
        temperature=0.0,
        response_format=QualityEvaluation,
    )
    return response.choices[0].message.parsed

# 3. El Endpoint (Donde Jira enviará los datos)
@app.post("/webhook/jira-ticket-created")
async def jira_listener(request: Request):
    try:
        # Simulamos recibir el payload JSON que envía Jira
        payload = await request.json()
        
        ticket_id = payload.get("issue_key", "UNKNOWN-1")
        titulo = payload.get("summary", "")
        descripcion = payload.get("description", "")
        
        texto_completo = f"{titulo} - {descripcion}"
        
        if len(texto_completo) < 10:
            raise HTTPException(status_code=400, detail="El ticket está vacío o es muy corto.")

        # 4. Invocamos al Juez IA (Capa 1)
        evaluacion = auditar_calidad_pbi(texto_completo)
        
        # 5. Lógica de Decisión (El Guardián del DoR)
        UMBRAL_ACEPTACION = 4.0
        
        if evaluacion.hu_q_score >= UMBRAL_ACEPTACION:
            estado = "✅ ACEPTADO (AI-Ready)"
            accion_jira = "Añadir etiqueta 'AI-Ready'. Mover a columna 'Listo para Refinamiento'."
        else:
            estado = "❌ RECHAZADO (AI-Rejected)"
            accion_jira = "Añadir comentario arrobando al creador exigiendo mejoras. Mover a columna 'Borrador'."

        # 6. Respuesta que le devolvemos a Jira
        respuesta_webhook = {
            "ticket_id": ticket_id,
            "estado_auditoria": estado,
            "nota_calidad": evaluacion.hu_q_score,
            "clasificacion": evaluacion.pbi_type,
            "comentario_sugerido_para_jira": f"🤖 Agile Tutor: {evaluacion.defect_reasoning}",
            "accion_automatizada": accion_jira
        }
        
        print(f"\n--- TICKET AUDITADO: {ticket_id} ---")
        print(f"Estado: {estado} | Nota: {evaluacion.hu_q_score}")
        print(f"Comentario: {evaluacion.defect_reasoning}")
        
        return respuesta_webhook

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- CAPA 3: EL BUCLE MLOPS (ESCUCHANDO A GITLAB/GITHUB) ---
class GitLabPayload(BaseModel):
    ticket_id: str
    lead_time_days: float
    archivos_modificados: int
    estimacion_original_sp: int
    dod_version_usado: str

@app.post("/webhook/gitlab-mr-merged")
async def gitlab_listener(payload: GitLabPayload):
    try:
        # 1. Analizar la Realidad vs La Predicción (Basado en Cohortes Estadísticas)
        # En la PoC, simulamos la media histórica de Lead Time (en días) para cada cohorte de Story Points.
        # En producción, esto se calcula dinámicamente haciendo un query al propio mlops_telemetry.csv
        medias_historicas_lead_time = {
            1: 1.5,   # 1 SP históricamente toma 1.5 días de Lead Time
            2: 2.5,
            3: 4.0,
            5: 7.0,
            8: 11.0,
            13: 18.0,
            21: 28.0
        }
        
        # Extraemos la media de la cohorte correspondiente, o usamos una heurística de fallback
        lead_time_promedio_cohorte = medias_historicas_lead_time.get(
            payload.estimacion_original_sp, 
            payload.estimacion_original_sp * 1.5
        )
        
        # 2. Detectar Shock Episódico (Umbral de Desviación)
        # Se declara "Shock" si el tiempo real excede la media histórica en más de un 50% (Margen Crítico)
        umbral_critico = lead_time_promedio_cohorte * 1.5 
        
        shock_detectado = False
        if payload.lead_time_days > umbral_critico:
            shock_detectado = True
            
        # 3. Guardar telemetría para el futuro reentrenamiento de LightGBM
        os.makedirs("data", exist_ok=True)
        ruta_csv = "data/mlops_telemetry.csv"
        archivo_existe = os.path.exists(ruta_csv)
        
        with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not archivo_existe:
                writer.writerow(["fecha", "ticket_id", "lead_time_days", "archivos_modificados", "sp_original", "dod_version", "shock_episodico"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                payload.ticket_id,
                payload.lead_time_days,
                payload.archivos_modificados,
                payload.estimacion_original_sp,
                payload.dod_version_usado,
                shock_detectado
            ])

        # 4. Generar la Alerta para el Arquitecto MLOps
        if shock_detectado:
            alerta = (f"🚨 ALERTA MLOPS: Posible Shock Episódico en {payload.ticket_id}. "
                      f"Se estimó en {payload.estimacion_original_sp} SP, pero tomó {payload.lead_time_days} días. "
                      f"DoD detectado: {payload.dod_version_usado}. Aislar estos datos del reentrenamiento base.")
        else:
            alerta = f"✅ Telemetría registrada para {payload.ticket_id}. Ejecución dentro de los márgenes normales."

        print(f"\n--- CI/CD MERGE DETECTADO: {payload.ticket_id} ---")
        print(alerta)
        
        return {
            "status": "Telemetría guardada",
            "alerta_mlops": alerta,
            "shock_episodico": shock_detectado
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Corre el servidor en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)