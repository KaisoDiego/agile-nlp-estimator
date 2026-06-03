# src/phase5_analysis/audit_team_accuracy.py

import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def get_fibonacci_suggestion(prediction):
    """Convierte un valor continuo al Fibonacci más cercano para poder comparar justamente."""
    fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    return min(fib_sequence, key=lambda x: abs(x - prediction))

def main():
    console.print("[bold cyan]🔍 Auditoría de Precisión: Modelo Base vs Modelo Ajustado por Equipo[/bold cyan]")

    # Buscar el archivo exportado desde la interfaz web (Streamlit)
    file_path = "agile_planning_session.csv"
    
    if not os.path.exists(file_path):
        console.print(f"[bold red]❌ No se encontró '{file_path}'.[/bold red]")
        console.print("[yellow]Instrucciones: Ve a tu aplicación Streamlit, evalúa al menos 5-10 PBIs, ajusta la columna 'Estimacion_Equipo' simulando el consenso final, y dale al botón 'Descargar Planificación para Jira'. Pon ese CSV en la raíz de tu proyecto.[/yellow]")
        return

    df = pd.read_csv(file_path)

    if 'Estimacion_Equipo' not in df.columns or 'SP_Crudos' not in df.columns:
        console.print("[bold red]❌ El CSV no tiene las columnas necesarias. Asegúrate de usar la última versión de la app.[/bold red]")
        return

    # 1. Transformar la predicción cruda base a Fibonacci para tener una comparación justa
    df['FIB_Base'] = df['SP_Crudos'].apply(get_fibonacci_suggestion)
    
    # 2. Calcular los Errores Absolutos (Qué tan lejos se equivocó la IA respecto al humano)
    df['Error_Base'] = np.abs(df['Estimacion_Equipo'] - df['FIB_Base'])
    df['Error_Ajustado'] = np.abs(df['Estimacion_Equipo'] - df['FIB_Sugerido'])

    # 3. Calcular el Error Absoluto Medio (MAE)
    mae_base = df['Error_Base'].mean()
    mae_ajustado = df['Error_Ajustado'].mean()
    mejora_porcentual = ((mae_base - mae_ajustado) / mae_base) * 100 if mae_base > 0 else 0

    # 4. Mostrar Resultados en Consola
    table = Table(title="Resultados del Experimento (A/B Testing de Fricción)")
    table.add_column("Métrica", style="cyan")
    table.add_column("Modelo Base (Solo LightGBM)", justify="center", style="red")
    table.add_column("Modelo Ajustado (Soporte LLM)", justify="center", style="green")

    table.add_row("Error Absoluto Medio (MAE)", f"{mae_base:.2f} SP", f"{mae_ajustado:.2f} SP")
    table.add_row("PBIs Estimados Perfectamente", 
                  f"{(df['Error_Base'] == 0).sum()} / {len(df)}", 
                  f"{(df['Error_Ajustado'] == 0).sum()} / {len(df)}")
    
    console.print(table)
    
    if mejora_porcentual > 0:
        console.print(f"\n✅ [bold green]ÉXITO ACADÉMICO:[/bold green] El contexto del equipo redujo el margen de error del algoritmo en un [bold]{mejora_porcentual:.1f}%[/bold].")
    else:
        console.print("\n⚠️ [bold yellow]NOTA:[/bold yellow] No hubo mejora. Asegúrate de haber modificado el entorno del equipo en la interfaz para que el multiplicador actuara.")

if __name__ == "__main__":
    main()