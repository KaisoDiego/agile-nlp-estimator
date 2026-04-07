# src/phase1_ingestion/smart_sampler.py

import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console

console = Console()

def snap_to_fibonacci_conservative(val):
    """Redondea conservadoramente al contenedor de Fibonacci."""
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    for fib in fibs:
        if val <= fib:
            return fib
    return fibs[-1]

def main():
    console.print("[bold cyan]🎯 Iniciando Muestreo Inteligente V2 (Mitigación de Amenazas)...[/bold cyan]")

    input_path = "data/02_processed/neodataset_limpio.parquet"
    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError:
        console.print(f"[bold red]❌ Error: No se encontró {input_path}[/bold red]")
        return

    # 1. Filtro de Longitud (Word Count Fallacy)
    df['word_count'] = df['user_story'].apply(lambda x: len(str(x).split()))
    df = df[(df['word_count'] >= 10) & (df['word_count'] <= 300)].copy()
    
    # 2. Aplicar Fibonacci temprano (Agrupar 15, 16 -> 21)
    df['target'] = df['target'].apply(snap_to_fibonacci_conservative)
    df = df[df['target'] <= 40]

    # 3. Mitigar Dominancia Cultural (Max 2500 registros por proyecto)
    console.print("[bold yellow]⚖️ Aplicando Límite de Representación (Max 2500/proyecto)...[/bold yellow]")
    
    # Solución limpia al DeprecationWarning de Pandas
    df_balanced = pd.concat([
        grupo.sample(min(len(grupo), 2500), random_state=42) 
        for _, grupo in df.groupby('project_id')
    ])

    # 4. Mitigar Escasez de Cola Larga (Preservar complejas)
    console.print("[bold magenta]🧬 Preservando todas las Historias Complejas (>=13 SP)...[/bold magenta]")
    raras = df_balanced[df_balanced['target'] >= 13].copy()
    comunes = df_balanced[df_balanced['target'] < 13].copy()

    faltan = 5000 - len(raras)

    # Red de seguridad: ¿Qué pasa si no hay suficientes?
    if faltan >= len(comunes):
        console.print(f"[yellow]⚠️ La piscina total es de {len(raras) + len(comunes)}. Usaremos todos los datos disponibles.[/yellow]")
        comunes_sample = comunes
    else:
        # Muestrear el resto estratificando las comunes
        _, comunes_sample = train_test_split(
            comunes, 
            test_size=faltan, 
            stratify=comunes['target'], 
            random_state=42
        )

    # Unir raras y comunes, y mezclar (shuffle)
    df_final = pd.concat([raras, comunes_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_final = df_final.drop(columns=['word_count'])

    # Guardar definitivo
    df_final.to_parquet(input_path, index=False)

    console.print(f"✅ ¡Éxito! Dataset de ÉLITE de {len(df_final)} registros guardado.")
    
    console.print("\n[bold green]📊 Distribución Fibonacci del Muestreo Final:[/bold green]")
    console.print(df_final['target'].value_counts().sort_index())

if __name__ == "__main__":
    main()