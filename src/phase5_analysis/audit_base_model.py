# src/phase5_analysis/audit_base_model.py

import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def get_complexity_bucket(sp):
    """Agrupa los Story Points en categorías para el análisis."""
    if sp <= 3:
        return "Pequeña (1-3 SP)"
    elif sp <= 8:
        return "Mediana (5-8 SP)"
    else:
        return "Épica/Compleja (13+ SP)"

def main():
    console.print(Panel.fit("[bold cyan]🔬 Auditoría Académica: Precisión del Modelo Base (Sin Equipo)[/bold cyan]"))

    # 1. Cargar Configuración
    with open("config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 2. Cargar Datos y Modelo
    console.print("[dim]Cargando datos y reconstruyendo el entorno de pruebas (Test Set)...[/dim]")
    df = pd.read_parquet(config["paths"]["scored_data"])
    embeddings = np.load(config["paths"]["embeddings"])
    artefactos = joblib.load(config["paths"]["model_save_path"])
    
    model = artefactos["model"]
    label_encoder = artefactos["label_encoder"]

    # --- REPLICAR EL FILTRO EXACTO DEL ENTRENAMIENTO ---
    mascara = df['target'] <= 40
    df = df[mascara].copy()
    embeddings = embeddings[mascara]
    
    df['pbi_type_encoded'] = label_encoder.transform(df['pbi_type'])
    scores = df[['hu_q_score', 'pbi_type_encoded']].values
    X = np.hstack((embeddings, scores))
    
    # Usamos target_clean si existe, si no, el target normal suavizado
    def snap_to_fibonacci_conservative(val):
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        for fib in fibs:
            if val <= fib: return fib
        return fibs[-1]
    y = df['target'].apply(snap_to_fibonacci_conservative).values

    # 3. Reconstruir el Test Set (Datos NUNCA VISTOS por el modelo)
    # Usamos el mismo random_state=42 que en train_lgbm.py para evitar Data Leakage
    _, X_test, _, y_test, _, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )

    # 4. Predicciones Base
    y_pred = np.maximum(1, model.predict(X_test)) # Mínimo 1 SP
    
    # Calcular Errores
    df_test = df_test.copy()
    df_test['Real_SP'] = y_test
    df_test['Pred_SP'] = y_pred
    df_test['Abs_Error'] = np.abs(df_test['Real_SP'] - df_test['Pred_SP'])
    df_test['Complexity'] = df_test['Real_SP'].apply(get_complexity_bucket)

    # --- RENDIMIENTO GLOBAL ---
    global_mae = mean_absolute_error(y_test, y_pred)
    global_mdae = median_absolute_error(y_test, y_pred)
    
    console.print("\n[bold yellow]📊 1. RENDIMIENTO GLOBAL (Test Set)[/bold yellow]")
    console.print(f"Error Absoluto Medio (MAE): [bold red]{global_mae:.2f} SP[/bold red]")
    console.print(f"Mediana del Error (MdAE): [bold red]{global_mdae:.2f} SP[/bold red]")

    # --- DESGLOSE POR TIPO DE TAREA ---
    console.print("\n[bold yellow]📊 2. ERROR POR TIPO DE PBI[/bold yellow]")
    table_type = Table(show_header=True, header_style="bold magenta")
    table_type.add_column("Tipo de Tarea")
    table_type.add_column("Volumen (Muestra)", justify="right")
    table_type.add_column("Error Promedio (MAE)", justify="right")

    for pbi_type in df_test['pbi_type'].unique():
        subset = df_test[df_test['pbi_type'] == pbi_type]
        mae_subset = subset['Abs_Error'].mean()
        table_type.add_row(str(pbi_type), str(len(subset)), f"{mae_subset:.2f} SP")
    console.print(table_type)

    # --- DESGLOSE POR COMPLEJIDAD (Long-Tail Blindness) ---
    console.print("\n[bold yellow]📊 3. ERROR POR COMPLEJIDAD (Demostración de Ceguera)[/bold yellow]")
    table_comp = Table(show_header=True, header_style="bold cyan")
    table_comp.add_column("Complejidad Real")
    table_comp.add_column("Volumen", justify="right")
    table_comp.add_column("Error Promedio (MAE)", justify="right", style="red")

    for comp in ["Pequeña (1-3 SP)", "Mediana (5-8 SP)", "Épica/Compleja (13+ SP)"]:
        subset = df_test[df_test['Complexity'] == comp]
        if not subset.empty:
            mae_subset = subset['Abs_Error'].mean()
            table_comp.add_row(comp, str(len(subset)), f"{mae_subset:.2f} SP")
    console.print(table_comp)
    
    # --- NUEVO: DISTRIBUCIÓN DE ACIERTOS (HIT RATE) ---
    console.print("\n[bold yellow]📊 4. DISTRIBUCIÓN DE ACIERTOS (Agile Hit Rate)[/bold yellow]")
    
    # 1. Convertir la predicción continua al Fibonacci más cercano
    def get_fibonacci_suggestion(val):
        fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        return min(fib_sequence, key=lambda x: abs(x - val))
    
    df_test['Pred_FIB'] = df_test['Pred_SP'].apply(get_fibonacci_suggestion)
    
    # 2. Calcular la distancia en "Cartas" o "Niveles"
    fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    fib_dict = {val: idx for idx, val in enumerate(fib_sequence)}
    
    def get_level_distance(real, pred):
        # Protegemos contra valores anómalos fuera de la secuencia
        if real not in fib_dict or pred not in fib_dict:
            return 99 
        return abs(fib_dict[real] - fib_dict[pred])
        
    df_test['Level_Distance'] = df_test.apply(lambda row: get_level_distance(row['Real_SP'], row['Pred_FIB']), axis=1)
    
    total = len(df_test)
    exactos = (df_test['Level_Distance'] == 0).sum()
    casi_exactos = (df_test['Level_Distance'] == 1).sum()
    graves = (df_test['Level_Distance'] >= 2).sum()
    
    table_hits = Table(show_header=True, header_style="bold green")
    table_hits.add_column("Precisión", style="cyan")
    table_hits.add_column("Volumen", justify="right")
    table_hits.add_column("Porcentaje", justify="right", style="magenta")
    table_hits.add_column("Interpretación Ágil", style="dim")
    
    table_hits.add_row("🎯 Acierto Exacto", str(exactos), f"{(exactos/total)*100:.1f}%", "La IA sugirió exactamente la misma carta que el equipo.")
    table_hits.add_row("🤏 Desvío de 1 Carta", str(casi_exactos), f"{(casi_exactos/total)*100:.1f}%", "Ej: IA dijo 5, Equipo dijo 8. Diferencia sana para debatir.")
    table_hits.add_row("🚨 Fallo Grave (2+ Cartas)", str(graves), f"{(graves/total)*100:.1f}%", "Ej: IA dijo 3, Equipo dijo 13. Requiere la Fricción del Equipo.")
    
    console.print(table_hits)

    # Exportar datos para gráficos si es necesario
    os.makedirs("reports", exist_ok=True)
    df_test.to_csv("reports/auditoria_modelo_base.csv", index=False)
    console.print("\n💾 [dim]Resultados detallados exportados a 'reports/auditoria_modelo_base.csv'[/dim]")

if __name__ == "__main__":
    main()