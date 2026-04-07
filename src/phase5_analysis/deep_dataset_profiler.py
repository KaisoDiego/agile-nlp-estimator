# src/phase5_analysis/deep_dataset_profiler.py

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    console.print("[bold cyan]🔬 Iniciando Radiografía Profunda del Dataset (Threats to Validity)...[/bold cyan]")

    file_path = "data/02_processed/neodataset_limpio.parquet"
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        console.print(f"[bold red]❌ No se encontró {file_path}[/bold red]")
        return

    total_records = len(df)
    console.print(f"📦 [dim]Analizando la muestra final de {total_records} registros.[/dim]\n")

    # ---------------------------------------------------------
    # 1. ANÁLISIS DE DOMINANCIA DE PROYECTOS (OVERFITTING CULTURAL)
    # ---------------------------------------------------------
    console.print("[bold yellow]1. Riesgo de Dominancia Cultural (Project Bias)[/bold yellow]")
    if 'project_id' in df.columns:
        project_counts = df['project_id'].value_counts(normalize=True) * 100
        
        table_proj = Table(show_header=True, header_style="bold magenta")
        table_proj.add_column("ID Proyecto")
        table_proj.add_column("Porcentaje en la Muestra", justify="right")
        
        for proj, pct in project_counts.items():
            color = "red" if pct > 50 else "green"
            table_proj.add_row(str(proj), f"[{color}]{pct:.1f}%[/{color}]")
        console.print(table_proj)
        
        if project_counts.iloc[0] > 60:
            console.print("⚠️ [red]PELIGRO:[/red] Un solo proyecto domina más del 60% del dataset. El modelo podría sobreajustarse a su jerga específica.")
        else:
            console.print("✅ [green]BUENO:[/green] Hay una distribución cultural decente entre los repositorios.")
    else:
        console.print("[dim]No se encontró la columna project_id para este análisis.[/dim]")
    print("\n")

    # ---------------------------------------------------------
    # 2. LA TRAMPA DEL CONTADOR DE PALABRAS (CORRELACIÓN DE LONGITUD)
    # ---------------------------------------------------------
    console.print("[bold yellow]2. Riesgo de Sesgo por Longitud (Word Count Fallacy)[/bold yellow]")
    df['word_count'] = df['user_story'].apply(lambda x: len(str(x).split()))
    
    correlacion = df['word_count'].corr(df['target'])
    
    console.print(f"Correlación de Pearson (Longitud vs Story Points): [bold cyan]{correlacion:.3f}[/bold cyan]")
    
    if correlacion > 0.6:
        console.print("⚠️ [red]PELIGRO (LO MALO):[/red] Correlación alta. El modelo podría estar adivinando los puntos solo por qué tan largo es el texto, ignorando la semántica técnica.")
    elif correlacion < 0.3:
        console.print("✅ [green]EXCELENTE (LO BUENO):[/green] Correlación baja. Esto demuestra que SBERT está forzado a leer y entender el significado de las palabras, porque un texto corto puede ser complejo y uno largo puede ser trivial.")
    else:
        console.print("ℹ️ [blue]NORMAL:[/blue] Existe una correlación moderada, típica en Ingeniería de Software (tareas complejas requieren más explicación).")
    print("\n")

    # ---------------------------------------------------------
    # 3. ESCASÉZ EN LA COLA LARGA (CLASES RARAS)
    # ---------------------------------------------------------
    console.print("[bold yellow]3. Viabilidad Predictiva en Historias Complejas (Sparsity)[/bold yellow]")
    
    historias_complejas = df[df['target'] >= 13]
    porcentaje_complejas = (len(historias_complejas) / total_records) * 100
    
    table_sparse = Table(show_header=True, header_style="bold magenta")
    table_sparse.add_column("Categoría Fibonacci")
    table_sparse.add_column("Cantidad de Ejemplos", justify="right")
    
    for sp in sorted(historias_complejas['target'].unique()):
        count = len(df[df['target'] == sp])
        table_sparse.add_row(f"{sp} SP", str(count))
        
    console.print(table_sparse)
    console.print(f"Total de historias de alta complejidad (>= 13 SP): {len(historias_complejas)} ({porcentaje_complejas:.1f}%)")
    
    if len(historias_complejas) < 50:
        console.print("⚠️ [red]PELIGRO (LO MALO):[/red] Tenemos muy pocos ejemplos de historias complejas. El modelo será muy malo prediciendo Épicas o historias de 13+ SP.")
    else:
        console.print("✅ [green]BUENO:[/green] Tenemos una masa crítica para que el modelo aprenda a identificar tareas de alta complejidad.")

if __name__ == "__main__":
    main()