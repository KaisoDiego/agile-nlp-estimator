# app/main.py

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

def judge_pbi(text: str):
    from src.phase2_router.prompts import SYSTEM_PROMPT_JUDGE
    response = client.beta.chat.completions.parse(
        model=config["llm_router"]["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
            {"role": "user", "content": f"Evalúa:\n{text}"}
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
st.markdown("Evalúa historias, estima Story Points y exporta el historial a tu herramienta de gestión (Jira/Trello).")

# --- INICIALIZAR MEMORIA (SESSION STATE) ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "PBI_Texto", "Clasificacion", "Nota_Calidad", "Justificacion", "SP_Crudos", "FIB_Sugerido", "Texto_Completo"
    ])

user_input = st.text_area("Texto del Product Backlog Item (PBI):", height=150, placeholder="Ej: Fix login button on mobile app...")

if st.button("Analizar y Estimar", type="primary"):
    if len(user_input.strip()) < 10:
        st.warning("⚠️ El PBI es muy corto para ser evaluado.")
    else:
        with st.spinner("🧠 1. El Juez IA está analizando la calidad y taxonomía..."):
            juez_result = judge_pbi(user_input)
            
        with st.spinner("🔢 2. SBERT está convirtiendo el texto a matemáticas..."):
            encoded_input = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = sbert_model(**encoded_input)
            embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
            
        with st.spinner("🎯 3. LightGBM está calculando el esfuerzo..."):
            type_encoded = label_encoder.transform([juez_result.pbi_type])[0]
            features = np.hstack((embedding, [[juez_result.hu_q_score, type_encoded]]))
            
            # Aseguramos que el mínimo sea 1 (No existe esfuerzo 0 en tareas válidas ágiles)
            raw_prediction = max(1.0, lgbm_model.predict(features)[0])
            fib_suggestion = get_fibonacci_suggestion(raw_prediction)

        # --- MOSTRAR RESULTADOS ---
        st.success("¡Análisis Completado!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Clasificación", juez_result.pbi_type)
        col2.metric("Nota de Calidad", f"{juez_result.hu_q_score} / 5.0")
        col3.metric("Estimación Sugerida (SP)", f"{fib_suggestion} SP", delta=f"{raw_prediction:.2f} crudos", delta_color="off")
            
        st.subheader("🗣️ Retroalimentación del Arquitecto")
        st.info(juez_result.defect_reasoning)

        # --- GUARDAR EN EL HISTORIAL ---
        nuevo_registro = pd.DataFrame([{
            "PBI_Texto": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "Clasificacion": juez_result.pbi_type,
            "Nota_Calidad": juez_result.hu_q_score,
            "Justificacion": juez_result.defect_reasoning,
            "SP_Crudos": round(float(raw_prediction), 2),
            "FIB_Sugerido": fib_suggestion,
            "Texto_Completo": user_input
        }])
        
        # Concatenar al inicio para que el más reciente aparezca arriba
        if st.session_state.history.empty:
            st.session_state.history = nuevo_registro
        else:
            st.session_state.history = pd.concat([nuevo_registro, st.session_state.history], ignore_index=True)

# --- 5. TABLA DE HISTORIAL Y EXPORTACIÓN ---
st.divider()
st.subheader("📋 Historial de la Sesión de Planning")

if not st.session_state.history.empty:
    # Mostrar tabla limpia en pantalla (ocultando el texto completo)
    st.dataframe(
        st.session_state.history.drop(columns=["Texto_Completo"]), 
        use_container_width=True,
        hide_index=True
    )
    
    # Botón para descargar a CSV
    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Planificación en CSV",
        data=csv,
        file_name='agile_planning_session.csv',
        mime='text/csv',
    )
else:
    st.caption("Aún no se han evaluado PBIs en esta sesión.")