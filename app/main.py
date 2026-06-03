# app/main.py

import sys

import streamlit as st
import os
import yaml
import torch
import numpy as np
import pandas as pd  # Importante para el manejo del historial y CSV
import joblib
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Agile PBI Tutor", page_icon="🚀", layout="wide")
load_dotenv(override=True)

with open("config/settings.yaml", "r") as file:
    config = yaml.safe_load(file)

# --- 2. CARGA DE MODELOS (CACHÉ PARA VELOCIDAD) ---
@st.cache_resource
def load_ml_models():
    # Cargar SBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["vectorizer"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sbert_model = AutoModel.from_pretrained(model_name).to(device)
    sbert_model.eval()
    
    # Cargar LightGBM
    artefactos = joblib.load(config["paths"]["model_save_path"])
    return tokenizer, sbert_model, device, artefactos["model"], artefactos["label_encoder"]

tokenizer, sbert_model, device, lgbm_model, label_encoder = load_ml_models()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 3. FUNCIONES DEL PIPELINE ---
class QualityEvaluation(BaseModel):
    pbi_type: str = Field(..., description="Clasificación: 'User Story', 'Technical Task', 'Bug', 'Spike'")
    hu_q_score: float = Field(..., description="Nota de 1.0 a 5.0")
    defect_reasoning: str = Field(..., description="Justificación corta")
    friction_reasoning: str = Field(..., description="Explicación de qué habilidades exige el PBI y cómo cruzan con el nivel actual del equipo")
    team_friction_multiplier: float = Field(..., description="Multiplicador de fricción basado en el equipo (0.5 a 2.0)")

def judge_pbi(text: str, team_context: str):
    from src.phase2_router.prompts import SYSTEM_PROMPT_APP_JUDGE
    response = client.beta.chat.completions.parse(
        model=config["llm_router"]["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_APP_JUDGE},
            {"role": "user", "content": f"Matriz del Equipo:\n{team_context}\n\nEvalúa este PBI:\n{text}"}
        ],
        temperature=0.0,
        response_format=QualityEvaluation,
    )
    return response.choices[0].message.parsed

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_fibonacci_suggestion(prediction):
    fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    suggestion = min(fib_sequence, key=lambda x: abs(x - prediction))
    return suggestion

# --- 4. INTERFAZ DE USUARIO Y MEMORIA ---
st.title("🚀 Asistente de Refinamiento Ágil (Tesis)")
# --- MATRIZ DE COMPETENCIAS (SIDEBAR) ---
# --- MATRIZ DE COMPETENCIAS Y FRICCIÓN (SIDEBAR) ---
st.sidebar.header("🛠️ Perfil de Fricción del Equipo")
st.sidebar.markdown(
    """
    **Escala de Fricción (1 al 4):**
    * **1:** Fluidez Máxima / Total Dominio
    * **2:** Fricción Menor / Manejable
    * **3:** Resistencia Alta / Fricción Significativa
    * **4:** Bloqueo Severo / Riesgo Crítico
    """
)
# --- NUEVO: VECTOR 4 (CONTROL DE SHOCKS EPISÓDICOS Y DOD) ---
st.sidebar.divider()
st.sidebar.subheader("🛡️ Políticas de Calidad (DoD)")
dod_version = st.sidebar.selectbox(
    "Versión actual del Definition of Done:",
    ["v1.0 (Básico - Compila y Pasa Unit Tests)", 
     "v2.0 (Estricto - 100% Cobertura y QA Manual)", 
     "v3.0 (Crítico - Pruebas E2E, SAST/DAST y Firma CISO)"],
    help="Si las reglas para cerrar un ticket se vuelven más estrictas, cambia esta versión. Evitará que el MLOps mezcle métricas históricas incompatibles."
)
# -----------------------------------------------------------
# Nuevo Marco de Factores de Fricción (Escala 1 a 4 para evitar el "3 seguro")
default_skills = pd.DataFrame({
    "Categoría": [
        "Arquitectura", "Arquitectura", "Arquitectura", 
        "DevOps", "DevOps", "DevOps", 
        "Cognitivo", "Cognitivo", "Cognitivo", 
        "Cumplimiento"
    ],
    "Factor de Fricción": [
        "Deuda Técnica y Código Legacy", 
        "Falta de Pruebas y Desconocimiento", 
        "Acoplamiento (Radio de Impacto)", 
        "Madurez CI/CD (Pipeline Frágil)", 
        "Dependencias de Otros Equipos", 
        "Divergencia de Entornos (Local vs Prod)", 
        "Novedad Algorítmica (I+D)", 
        "Barrera de Dominio del Negocio", 
        "Interrupciones (Context-Switching)", 
        "Riesgo de Seguridad y Privacidad"
    ],
    "Nivel (1-4)": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] # 2 = Fricción Menor (Default)
})

# El data_editor permite al usuario cambiar los niveles del 1 al 4
equipo_df = st.sidebar.data_editor(
    default_skills, 
    hide_index=True, 
    use_container_width=True,
    column_config={
        "Nivel (1-4)": st.column_config.NumberColumn(
            "Nivel (1-4)", min_value=1, max_value=4, step=1
        )
    }
)
team_context_str = equipo_df.to_string(index=False)
st.markdown("Evalúa historias, estima Story Points y exporta el historial a tu herramienta de gestión (Jira/Trello).")

# --- INICIALIZAR MEMORIA (SESSION STATE) ---
# --- INICIALIZAR MEMORIA (SESSION STATE) ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "PBI_Texto", 
        "Clasificacion", 
        "Nota_Calidad", 
        "Justif_Calidad",  # Nueva columna 1
        "Justif_Equipo",   # Nueva columna 2
        "Friccion_Equipo", 
        "SP_Crudos", 
        "SP_Ajustados", 
        "FIB_Sugerido", 
        "Estimacion_Equipo", 
        "Texto_Completo"
    ])

user_input = st.text_area("Texto del Product Backlog Item (PBI):", height=150, placeholder="Ej: Fix login button on mobile app...")

if st.button("Analizar y Estimar", type="primary"):
    if len(user_input.strip()) < 10:
        st.warning("⚠️ El PBI es muy corto para ser evaluado.")
    else:
        with st.spinner("🧠 1. El Juez IA está analizando la calidad y el contexto del equipo..."):
            juez_result = judge_pbi(user_input, team_context_str) # <-- Pasamos la matriz aquí
            
        with st.spinner("🔢 2. SBERT está convirtiendo el texto a matemáticas..."):
            encoded_input = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = sbert_model(**encoded_input)
            embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
            
        with st.spinner("🎯 3. Calculando el esfuerzo ajustado..."):
            type_encoded = label_encoder.transform([juez_result.pbi_type])[0]
            features = np.hstack((embedding, [[juez_result.hu_q_score, type_encoded]]))
            
            # 1. Estimación del mercado base (LightGBM)
            raw_prediction = max(1.0, lgbm_model.predict(features)[0])
            
            # 2. MITIGACIÓN VECTOR 2: ESCALADO NO LINEAL (COMO SUGIERE LA AUDITORÍA)
            # Implementamos un factor de des-economía de escala (B > 1.0) para la Cola Larga.
            # Si la tarea es pequeña, el impacto es lineal. Si es grande (>=13 SP), es exponencial.
            
            base_multiplier = juez_result.team_friction_multiplier
            
            if raw_prediction >= 13:
                # Factor de exponente para simular crecimiento exponencial de complejidad
                # según la fórmula Effort = A * Size^B
                complexity_exponent = 1.25 
                adjusted_multiplier = base_multiplier ** complexity_exponent
                adjusted_prediction = raw_prediction * adjusted_multiplier
                is_long_tail = True
            else:
                adjusted_prediction = raw_prediction * base_multiplier
                is_long_tail = False
                
            fib_suggestion = get_fibonacci_suggestion(adjusted_prediction)

        # --- MOSTRAR RESULTADOS (MODIFICADO PARA VECTOR 1: DOBLE ANCLAJE CIEGO) ---
        st.success("¡Análisis Completado! El Juez IA ha emitido su veredicto en secreto.")
        
        if is_long_tail and juez_result.team_friction_multiplier > 1.2:
            st.warning("🚨 **ALERTA DE COLA LARGA:** Esta tarea es intrínsecamente compleja y la fricción del equipo está escalando el esfuerzo de forma exponencial. Procede con extrema cautela.")
        
        # Ocultamos el resultado para forzar al equipo a debatir primero
        with st.expander("🤖 Revelar Estimación de la IA (Abrir SOLO después de votar)", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Clasificación", juez_result.pbi_type)
            col2.metric("Nota de Calidad", f"{juez_result.hu_q_score} / 5.0")
            col3.metric("Fricción Equipo", f"x{juez_result.team_friction_multiplier}")
            col4.metric("Estimación Sugerida", f"{fib_suggestion} SP", delta=f"{adjusted_prediction:.2f} crudos", delta_color="off")
                
            st.subheader("🗣️ Retroalimentación del Arquitecto")
            st.info(f"**Sobre la Calidad:** {juez_result.defect_reasoning}")
            st.warning(f"**Sobre el Equipo:** {juez_result.friction_reasoning}")

        # --- GUARDAR EN EL HISTORIAL (MODIFICADO PARA VECTOR 4: DOD) ---
        nuevo_registro = pd.DataFrame([{
            "PBI_Texto": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "DoD_Version": dod_version.split(" ")[0], # <-- NUEVO: Guardamos v1.0, v2.0, etc.
            "Clasificacion": juez_result.pbi_type,
            "Nota_Calidad": juez_result.hu_q_score,
            "Justif_Calidad": juez_result.defect_reasoning,
            "Justif_Equipo": juez_result.friction_reasoning,
            "Friccion_Equipo": juez_result.team_friction_multiplier,
            "SP_Crudos": round(float(raw_prediction), 2),
            "SP_Ajustados": round(float(adjusted_prediction), 2),
            "FIB_Sugerido": fib_suggestion,
            "Estimacion_Equipo": fib_suggestion,
            "Texto_Completo": user_input
        }])
        
        # Concatenar al inicio para que el más reciente aparezca arriba
        if st.session_state.history.empty:
            st.session_state.history = nuevo_registro
        else:
            st.session_state.history = pd.concat([nuevo_registro, st.session_state.history], ignore_index=True)

# --- 5. TABLA DE HISTORIAL Y EXPORTACIÓN ---
st.divider()
st.subheader("📋 Historial y Calibración (Active Learning)")
st.markdown("¿El equipo difiere de la IA? Haz doble clic en la columna **Estimacion_Equipo** para corregirla.")

if not st.session_state.history.empty:
    columnas_visibles = st.session_state.history.drop(columns=["Texto_Completo"])
    
    edited_df = st.data_editor(
        columnas_visibles, 
        use_container_width=True,
        hide_index=True,
        # Añadimos las nuevas columnas a la lista de deshabilitadas para que no se editen por error
        disabled=[
            "PBI_Texto", "Clasificacion", "Nota_Calidad", "Justif_Calidad", 
            "Justif_Equipo", "Friccion_Equipo", "SP_Crudos", "SP_Ajustados", "FIB_Sugerido"
        ],
        column_config={
            "Justif_Calidad": st.column_config.TextColumn("Justificación Calidad", width="medium"),
            "Justif_Equipo": st.column_config.TextColumn("Justificación Equipo", width="medium"),
        }
    )
    
    # Actualizamos la memoria interna con lo que el usuario haya editado en la UI
    st.session_state.history["Estimacion_Equipo"] = edited_df["Estimacion_Equipo"]
    
    col1, col2 = st.columns(2)
    with col1:
        # Botón 1: Exportación normal a Jira
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Planificación para Jira",
            data=csv,
            file_name='agile_planning_session.csv',
            mime='text/csv',
        )
        
    with col2:
        # Botón 2: El Bucle de Calibración
        if st.button("💾 Guardar Feedback para Reentrenamiento", type="secondary"):
            # Extraemos solo lo necesario para reentrenar a SBERT/LightGBM en el futuro
            reentrenamiento_df = st.session_state.history[["Texto_Completo", "Estimacion_Equipo", "Clasificacion"]]
            
            # Guardamos en la carpeta data
            os.makedirs("data", exist_ok=True)
            ruta_calibracion = "data/calibracion_local.csv"
            
            # Si el archivo ya existe, lo añade al final (append), si no, lo crea
            if os.path.exists(ruta_calibracion):
                reentrenamiento_df.to_csv(ruta_calibracion, mode='a', header=False, index=False)
            else:
                reentrenamiento_df.to_csv(ruta_calibracion, index=False)
                
            st.success("✅ ¡Feedback capturado! Los datos se han guardado en `data/calibracion_local.csv`.")
else:
    st.caption("Aún no se han evaluado PBIs en esta sesión.")