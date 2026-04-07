# src/phase5_analysis/evaluate_errors.py

import os
import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    console.print("[bold cyan]🔍 Iniciando Análisis de Errores y Gráficos Académicos...[/bold cyan]")

    # 1. Cargar Configuración y Datos
    with open("config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)

    df = pd.read_parquet(config["paths"]["scored_data"])
    embeddings = np.load(config["paths"]["embeddings"])
    artefactos = joblib.load(config["paths"]["model_save_path"])
    
    model = artefactos["model"]
    label_encoder = artefactos["label_encoder"]

    # 2. Reconstruir las características exactas
    # --- FILTRO APLICADO PARA LA EVALUACIÓN ---
    mascara = df['target'] <= 40
    df = df[mascara].copy()
    embeddings = embeddings[mascara]
    # ------------------------------------------
    
    df['pbi_type_encoded'] = label_encoder.transform(df['pbi_type'])
    scores = df[['hu_q_score', 'pbi_type_encoded']].values
    X = np.hstack((embeddings, scores))
    y_real = df['target'].values

    # 3. Hacer predicciones sobre TODO el dataset
    y_pred = np.maximum(0, model.predict(X))
    
    # 4. Calcular el Error de cada historia
    df['Predicted_SP'] = y_pred
    df['Absolute_Error'] = np.abs(df['target'] - df['Predicted_SP'])

    # --- GENERACIÓN DE GRÁFICOS PARA LA TESIS ---
    os.makedirs("reports/figures", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Real vs Predicho (Matriz de Dispersión)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_real, y_pred, alpha=0.4, color='purple')
    plt.plot([0, max(y_real)], [0, max(y_real)], '--', color='red', linewidth=2) # Línea de perfección
    plt.title("Estimación Real vs Predicción de la IA (Story Points)")
    plt.xlabel("Story Points Reales (Equipo)")
    plt.ylabel("Predicción del Modelo (IA)")
    plt.savefig("reports/figures/real_vs_pred.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico 2: Importancia de Variables (¿Qué mira la IA?)
    # Nombrar las 770 columnas (768 de SBERT + Nota + Tipo)
    feature_names = [f"Emb_{i}" for i in range(768)] + ["Nota_Calidad", "Tipo_PBI"]
    importancias = model.feature_importances_
    
    # Obtener el Top 10 de variables más importantes
    indices_top = np.argsort(importancias)[-10:]
    nombres_top = [feature_names[i] for i in indices_top]
    valores_top = importancias[indices_top]

    plt.figure(figsize=(10, 6))
    plt.barh(nombres_top, valores_top, color='teal')
    plt.title("Top 10 Variables con Mayor Peso en la Decisión (LightGBM)")
    plt.xlabel("Importancia (Nodos del Árbol)")
    plt.savefig("reports/figures/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print("✅ [green]Gráficos guardados en la carpeta 'reports/figures/'[/green]")

    # --- ANÁLISIS DE CONFUSIÓN (Los peores errores) ---
    df_errores = df.sort_values(by='Absolute_Error', ascending=False).head(5)
    
    console.print("\n[bold red]🚨 TOP 5 PBIs QUE CONFUNDIERON AL ALGORITMO (Análisis de Anomalías)[/bold red]")
    
    for idx, row in df_errores.iterrows():
        table = Table(show_header=False, box=None)
        table.add_row(f"[bold magenta]Real: {row['target']} SP[/bold magenta] | [bold cyan]IA dijo: {row['Predicted_SP']:.1f} SP[/bold cyan] | [bold red]Error: {row['Absolute_Error']:.1f} SP[/bold red]")
        console.print(table)
        console.print(f"[dim]Tipo: {row['pbi_type']} | Nota: {row['hu_q_score']}[/dim]")
        console.print(f"[white]{row['user_story'][:300]}...[/white]")
        console.print("-" * 80)

if __name__ == "__main__":
    main()