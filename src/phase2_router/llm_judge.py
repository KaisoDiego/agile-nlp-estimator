# src/phase2_router/llm_judge.py

import os
import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import time
from prompts import SYSTEM_PROMPT_BATCH_JUDGE
# --- NUEVOS IMPORTS DE RICH ---
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel

# Inicializar la consola enriquecida
console = Console()

# 1. Cargar Variables de Entorno y Configuración
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("config/settings.yaml", "r") as file:
    config = yaml.safe_load(file)

# 2. Definir la Estructura Exacta que OpenAI debe devolver
class QualityEvaluation(BaseModel):
    pbi_type: str = Field(..., description="Clasificación estricta: 'User Story', 'Technical Task', 'Bug', o 'Spike'")
    hu_q_score: float = Field(..., ge=1.0, le=5.0, description="Nota de calidad (1.0 a 5.0)")
    defect_reasoning: str = Field(..., description="Justificación técnica corta de la nota y clasificación")
def evaluate_pbi(text: str, model_name: str, temperature: float) -> QualityEvaluation:
    """Envía el texto a OpenAI y fuerza una respuesta estructurada (JSON)."""
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BATCH_JUDGE},
            {"role": "user", "content": f"Evalúa este PBI:\n{text}"}
        ],
        temperature=temperature,
        response_format=QualityEvaluation,
    )
    return response.choices[0].message.parsed

def main():
    # MODO PILOTO: Cambia a False para procesar los datos reales
    PILOT_MODE = False
    
    # Encabezado visual bonito
    console.print(Panel.fit("[bold cyan]🚀 Iniciando el Enrutador Semántico (Fase 2) con IA...[/bold cyan]", border_style="cyan"))
    
    # Leer rutas y variables desde el config.yaml
    input_path = config["paths"]["processed_data"]
    output_path = config["paths"]["scored_data"]
    checkpoint_path = output_path.replace(".parquet", "_checkpoint.parquet")
    
    model_name = config["llm_router"]["model_name"]
    temp = config["llm_router"]["temperature"]
    batch_save_interval = config["llm_router"]["batch_save_interval"]
    
    # Verificar si el archivo limpio existe
    if not os.path.exists(input_path):
        console.print(f"[bold red]❌ Error: No se encontró el dataset en {input_path}[/bold red]")
        return

    # SISTEMA DE RECUPERACIÓN ANTE DESASTRES (CHECKPOINT)
    if os.path.exists(checkpoint_path) and not PILOT_MODE:
        console.print(f"[bold yellow]🔄 ¡Checkpoint encontrado![/bold yellow] Retomando evaluación previa desde [dim]{checkpoint_path}[/dim]...")
        df = pd.read_parquet(checkpoint_path)
    else:
        console.print("[dim]📦 Cargando dataset limpio desde cero...[/dim]")
        df = pd.read_parquet(input_path)
        
        # Reducción estratégica para la tesis (3000 registros)
        """ df = df.sample(3000, random_state=42).copy().reset_index(drop=True) """
        
        # Crear nuevas columnas vacías si no existen
        for col in ["pbi_type", "hu_q_score", "defect_reasoning", "is_atomic"]:
            if col not in df.columns:
                df[col] = None

    if PILOT_MODE:
        console.print("[bold red]⚠️ MODO PILOTO ACTIVADO:[/bold red] Evaluando solo 5 registros al azar.")
        df = df.sample(5).copy()
        df.reset_index(drop=True, inplace=True)
    
    # Filtrar solo los registros que aún no han sido evaluados
    pendientes = df[df["pbi_type"].isnull()]
    total_pendientes = len(pendientes)
    
    if total_pendientes == 0:
        console.print("[bold green]✅ Todos los PBIs ya han sido evaluados. No hay nada que hacer.[/bold green]")
        return

    console.print(f"[bold yellow]📊 PBIs pendientes de evaluar:[/bold yellow] {total_pendientes} de {len(df)}")
    console.print("-" * 60)

    procesados_en_esta_sesion = 0
    errores_consecutivos = 0

    # --- CONFIGURACIÓN DE LA BARRA DE PROGRESO RICH ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("• ETA:"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        # Añadir la tarea a la barra
        task_eval = progress.add_task("[cyan]Analizando PBIs...", total=total_pendientes)

        for index, row in pendientes.iterrows():
            texto_pbi = row["user_story"]
            
            try:
                resultado = evaluate_pbi(texto_pbi, model_name, temp)
                
                # Guardar resultados
                df.at[index, "pbi_type"] = resultado.pbi_type
                df.at[index, "hu_q_score"] = resultado.hu_q_score
                df.at[index, "defect_reasoning"] = resultado.defect_reasoning
                
                # Imprimir el log de éxito *por encima* de la barra de progreso
                color_tipo = "green" if resultado.pbi_type == "User Story" else "blue" if resultado.pbi_type == "Technical Task" else "red" if resultado.pbi_type == "Bug" else "magenta"
                progress.console.print(f"[dim]ID {index}:[/dim] [{color_tipo}]■ {resultado.pbi_type}[/{color_tipo}] | ⭐ Nota: [bold]{resultado.hu_q_score}[/bold]/5.0")
                
                errores_consecutivos = 0
                procesados_en_esta_sesion += 1
                
                # Avanzar la barra
                progress.advance(task_eval)
                
                # AUTO-GUARDADO (CHECKPOINT)
                if not PILOT_MODE and procesados_en_esta_sesion % batch_save_interval == 0:
                    progress.console.print(f"[bold magenta]💾 Guardando checkpoint temporal de seguridad ({procesados_en_esta_sesion} procesados)...[/bold magenta]")
                    df.to_parquet(checkpoint_path, index=False)

            except Exception as e:
                progress.console.print(f"[bold red]❌ Error al evaluar el PBI {index}: {e}[/bold red]")
                errores_consecutivos += 1
                
                if errores_consecutivos >= 3:
                    progress.console.print(Panel.fit("[bold red]⚠️ 3 Errores consecutivos detectados (Caída de API/Rate Limit).\n🛑 Deteniendo ejecución por seguridad.[/bold red]"))
                    break
                time.sleep(2) 

    # GUARDADO FINAL
    console.print("\n" + "="*60)
    if PILOT_MODE:
        ruta_piloto = "data/02_processed/piloto_revision.csv"
        df.to_csv(ruta_piloto, index=False, encoding="utf-8-sig")
        console.print(f"💾 [bold green]¡Piloto guardado en {ruta_piloto}![/bold green]")
    else:
        console.print(f"💾 [bold cyan]Guardando Dataset Final en {output_path}...[/bold cyan]")
        df.to_parquet(output_path, index=False)
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
        console.print(Panel.fit(f"[bold green]✅ ¡Fase 2 Completada Exitosamente![/bold green]\nSe evaluaron {procesados_en_esta_sesion} PBIs.", border_style="green"))

if __name__ == "__main__":
    main()