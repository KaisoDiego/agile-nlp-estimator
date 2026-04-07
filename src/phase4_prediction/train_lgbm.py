# src/phase4_prediction/train_lgbm.py

import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import LabelEncoder

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def main():
    console.print(Panel.fit("[bold cyan]🚀 Iniciando Fase 4: Entrenamiento del Motor Predictivo (Optimizado)[/bold cyan]", border_style="cyan"))

    # 1. Cargar Configuración
    with open("config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)

    data_path = config["paths"]["scored_data"]
    embeddings_path = config["paths"]["embeddings"]
    model_save_path = config["paths"]["model_save_path"]

    # 2. Cargar los Datos (El "Puente" entre Fase 2 y Fase 3)
    console.print("[dim]📦 Cargando dataset puntuado y matriz matemática...[/dim]")
    df = pd.read_parquet(data_path)
    embeddings = np.load(embeddings_path)

    # --- A) FILTRO DE ANOMALÍAS (Label Filtering) ---
    # Solo entrenaremos con historias menores o iguales a 40 SP
    mascara = df['target'] <= 40
    df = df[mascara].copy()
    embeddings = embeddings[mascara]
    console.print(f"[bold red]🧹 Purga aplicada: Ignorando Épicos y anomalías (>40 SP). Datos útiles: {len(df)}[/bold red]")
    
    # Verificar que el número de filas coincida
    if len(df) != embeddings.shape[0]:
        console.print(f"[bold red]❌ Error: El dataset tiene {len(df)} filas, pero los embeddings tienen {embeddings.shape[0]}. Deben ser idénticos.[/bold red]")
        return

    # --- B) TARGET SMOOTHING (Suavizado de Objetivo) ---
    # Redondeamos "Horas Ideales" humanas hacia el contenedor Fibonacci superior más cercano
    console.print("[bold yellow]🔧 Suavizando estimaciones lineales a contenedores de Fibonacci (Target Smoothing)...[/bold yellow]")
    
    def snap_to_fibonacci_conservative(val):
        """Redondea conservadoramente hacia el valor de Fibonacci igual o superior más cercano."""
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        for fib in fibs:
            if val <= fib:
                return fib
        return fibs[-1]

    df['target_clean'] = df['target'].apply(snap_to_fibonacci_conservative)

    # 3. Ingeniería de Características (Feature Engineering)
    console.print("[bold cyan]⚙️ Fusionando variables categóricas, notas y vectores...[/bold cyan]")
    
    # Convertir el tipo de PBI a números (ej. Bug=0, Spike=1, User Story=2...)
    le_type = LabelEncoder()
    df['pbi_type_encoded'] = le_type.fit_transform(df['pbi_type'])

    # Extraer las variables extra creadas por el Juez IA
    scores = df[['hu_q_score', 'pbi_type_encoded']].values

    # Unir la matriz de SE-BERT con las notas del Juez
    X = np.hstack((embeddings, scores))
    
    # IMPORTANTE: Ahora la IA aprenderá de la variable limpia (contenedores), no de la sucia
    y = df['target_clean'].values

    # 4. Dividir en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    console.print(f"📊 Distribución: [bold]Entrenamiento:[/bold] {len(X_train)} | [bold]Prueba (Invisibles):[/bold] {len(X_test)}")

    # 5. Entrenar el Modelo LightGBM
    console.print("[bold magenta]🧠 Entrenando modelo de potenciación del gradiente (LightGBM)...[/bold magenta]")
    
    # Hiperparámetros base (optimizados para regresión rápida y evitar warnings)
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)

    # 6. Evaluación Científica Insesgada (MAE y MdAE)
    console.print("[bold green]🎯 Evaluando métricas insesgadas (MAE y MdAE)...[/bold green]")
    y_pred = model.predict(X_test)

    # Evitar predicciones menores a 1 SP (no existe esfuerzo 0 en Fibonacci para PBI aceptados)
    y_pred = np.maximum(y_pred, 1)

    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Crear una tabla hermosa para los resultados
    table = Table(title="Métricas de Rendimiento del Modelo (Optimizado)")
    table.add_column("Métrica", style="cyan", no_wrap=True)
    table.add_column("Valor", style="magenta")
    table.add_column("Interpretación Académica", style="green")

    table.add_row("MAE (Error Absoluto Medio)", f"{mae:.2f} SP", "Error absoluto promedio.")
    table.add_row("MdAE (Mediana del Error)", f"{mdae:.2f} SP", "Resistente a atípicos. Error central del modelo.")
    table.add_row("R² (Coef. de Determinación)", f"{r2:.4f}", "Varianza explicada sobre los contenedores Fibonacci.")

    console.print(table)

    # 7. Guardado para Producción
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Guardamos el modelo y el decodificador de etiquetas juntos
    artefactos = {
        "model": model,
        "label_encoder": le_type
    }
    joblib.dump(artefactos, model_save_path)
    
    console.print(f"💾 [bold green]Modelo empacado y guardado en {model_save_path}[/bold green]")

if __name__ == "__main__":
    main()