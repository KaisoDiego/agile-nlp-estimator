# src/phase5_analysis/audit_projects.py

import pandas as pd
import os
import glob
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    console.print("[bold cyan]🔍 Iniciando Auditoría de Escalas Múltiples (Leyendo Archivos Crudos)...[/bold cyan]")

    # 1. Usar tu misma configuración de la Fase 1
    with open("config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    raw_folder = config["paths"]["raw_data_folder"]

    # 2. Buscar todos los CSVs igual que en tu ingest.py
    all_files = glob.glob(os.path.join(raw_folder, "*.csv"))
    if not all_files:
        console.print(f"[bold red]❌ Error: No se encontraron archivos CSV en {raw_folder}[/bold red]")
        return

    console.print(f"[dim]📁 Se detectaron {len(all_files)} archivos de proyecto. Analizando huellas dactilares...[/dim]")

    project_stats = []

    # 3. Iterar sobre cada archivo (Proyecto)
    for filepath in all_files:
        # El nombre del archivo es el ID del proyecto (ej: '278964.csv' -> '278964')
        project_id = os.path.basename(filepath).replace('.csv', '')

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            console.print(f"[yellow]⚠️ Error leyendo el proyecto {project_id}: {e}[/yellow]")
            continue

        target_column = 'storypoints'
        
        # Verificar si la columna existe en este CSV
        if target_column not in df.columns:
            continue

        # Limpiar nulos para no falsear los conteos
        puntos_limpios = df[target_column].dropna()
        total_pbis = len(puntos_limpios)

        # Solo auditar proyectos que tengan un tamaño estadísticamente relevante
        if total_pbis < 50: 
            continue

        # 4. Extraer la Huella Digital (Estadísticas)
        top_values = puntos_limpios.value_counts().head(5).index.tolist()
        max_sp = puntos_limpios.max()
        median_sp = puntos_limpios.median()

        project_stats.append({
            'Proyecto ID': str(project_id),
            'PBIs Valuados': total_pbis,
            'Valores Más Comunes (Top 5)': str([round(v, 1) for v in top_values]),
            'SP Máximo': max_sp,
            'Mediana': median_sp
        })

    # 5. Generar Tabla de Resultados
    df_stats = pd.DataFrame(project_stats)
    if not df_stats.empty:
        # Ordenar por los proyectos más grandes primero
        df_stats = df_stats.sort_values(by='PBIs Valuados', ascending=False)

        table = Table(title="Auditoría de Escalas (Huellas Digitales por Proyecto)")
        table.add_column("Proyecto ID (CSV)", style="cyan")
        table.add_column("PBIs Valuados", justify="right")
        table.add_column("Valores Más Comunes", style="magenta")
        table.add_column("Max SP", justify="right")
        table.add_column("Mediana", justify="right")

        for _, row in df_stats.iterrows():
            table.add_row(
                row['Proyecto ID'],
                str(row['PBIs Valuados']),
                row['Valores Más Comunes (Top 5)'],
                str(round(row['SP Máximo'], 1)),
                str(round(row['Mediana'], 1))
            )

        console.print(table)
        
        # Guardar evidencia para tu tesis
        os.makedirs("reports", exist_ok=True)
        df_stats.to_csv("reports/auditoria_proyectos_crudo.csv", index=False)
        console.print("\n💾 [green]Evidencia exportada a 'reports/auditoria_proyectos_crudo.csv'[/green]")
    else:
        console.print("[yellow]⚠️ No se encontraron proyectos con suficientes datos para analizar.[/yellow]")

if __name__ == "__main__":
    main()