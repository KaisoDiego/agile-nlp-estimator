📁 Modelo_Tesis_Estimacion/
│
├── 📁 data/                      # 🗄️ El almacén intocable (Ignorado en GitHub)
│   ├── 📁 01_raw/                # Datos crudos de GitLab/Jira (CSVs originales)
│   ├── 📁 02_processed/          # Dataset limpio (neodataset_limpio.parquet)
│   └── 📁 03_embeddings/         # Vectores matemáticos (matrices .npy de SBERT)
│
├── 📁 models/                    # 🧠 Modelos guardados
│   └── lightgbm_pbi_model.pkl    # El modelo final entrenado y empaquetado
│
├── 📁 notebooks/                 # 📓 Entornos de experimentación (Jupyter)
│   ├── 01_eda_y_limpieza.ipynb   # Análisis exploratorio de datos
│   └── 02_pruebas_lightgbm.ipynb # Pruebas iniciales de los árboles de decisión
│
├── 📁 src/                       # ⚙️ El código fuente de tu aplicación (El Motor)
│   ├── __init__.py
│   │
│   ├── 📁 phase1_ingestion/      # FASE 1: Limpieza
│   │   ├── ingest.py             # Carga los CSVs
│   │   └── clean_text.py         # Limpia caracteres pero respeta el contenido
│   │
│   ├── 📁 phase2_router/         # FASE 2: El Juez LLM y Taxonomía
│   │   ├── llm_judge.py          # Conexión a OpenAI y ejecución de clasificación
│   │   └── prompts.py            # Tu prompt maestro (INVEST, CIDEM, Bug, Spike)
│   │
│   ├── 📁 phase3_vectorization/  # FASE 3: Traducción Matemática
│   │   └── vectorize_sbert.py    # Convierte texto a matrices con se-bert
│   │
│   └── 📁 phase4_prediction/     # FASE 4: El Modelo LightGBM
│       ├── train_lgbm.py         # Script para entrenar el modelo con las 3 ramas
│       └── predict.py            # Script para hacer predicciones de nuevos PBIs
│
├── 📁 app/                       # 🌐 El Producto Final (Lo que ve el usuario)
│   ├── api.py                    # Microservicio FastAPI (Recibe peticiones)
│   └── ui_streamlit.py           # Interfaz gráfica web interactiva
│
├── .env                          # 🔑 Tus llaves secretas (OPENAI_API_KEY)
├── .gitignore                    # Evita subir datos pesados o llaves a GitHub
├── requirements.txt              # Dependencias (pandas, lightgbm, openai, fastapi...)
└── README.md                     # Manual de instrucciones de tu proyecto