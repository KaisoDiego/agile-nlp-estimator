# src/phase3_vectorization/vectorize_sbert.py

import os
import yaml
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel

# Inicializar consola profesional
console = Console()

def get_device():
    """Detecta automáticamente si hay una Tarjeta Gráfica (GPU) disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def mean_pooling(model_output, attention_mask):
    """Comprime la salida neuronal en un solo vector matemático por historia."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    console.print(Panel.fit("[bold cyan]🧠 Iniciando Fase 3: Traducción Matemática con SE-BERT[/bold cyan]", border_style="cyan"))

    # 1. Cargar configuración
    with open("config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)

    # ⚠️ IMPORTANTE: Leemos el dataset PUNTUADO de la Fase 2
    input_path = config["paths"]["scored_data"]
    output_path = config["paths"]["embeddings"]
    model_name = config["vectorizer"]["model_name"]
    batch_size = config["vectorizer"]["batch_size"]

    # 2. Verificar datos
    if not os.path.exists(input_path):
        console.print(f"[bold red]❌ Error: No se encontró el dataset evaluado en {input_path}[/bold red]")
        return

    df = pd.read_parquet(input_path)
    textos = df['user_story'].tolist()
    console.print(f"[dim]📦 Cargados {len(textos)} PBIs listos para vectorización matemática.[/dim]")

    # 3. Preparar el Modelo
    device = get_device()
    console.print(f"[bold yellow]⚙️ Cargando modelo '{model_name}' en [{device}]...[/bold yellow]")
    
    # Descarga el modelo de HuggingFace (solo la primera vez)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval() # Modo inferencia estricta

    # 4. Procesamiento
    all_embeddings = []
    
    # Interfaz de progreso
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
        
        task_vec = progress.add_task("[cyan]Convirtiendo texto a matemáticas...", total=len(textos))
        
        with torch.no_grad():
            for i in range(0, len(textos), batch_size):
                batch_texts = textos[i : i + batch_size]
                
                # Tokenizar
                encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
                
                # Pasar por la red neuronal
                model_output = model(**encoded_input)
                
                # Agrupar y mover a memoria RAM normal
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                all_embeddings.append(sentence_embeddings.cpu().numpy())
                
                # Avanzar UI
                progress.advance(task_vec, advance=len(batch_texts))

    # 5. Guardado de Matriz
    final_embeddings = np.vstack(all_embeddings)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, final_embeddings)
    
    console.print("\n" + "="*60)
    resumen = (f"[bold green]✅ ¡Fase 3 Completada![/bold green]\n"
               f"Matriz guardada en: [dim]{output_path}[/dim]\n"
               f"Forma de la matriz: [bold yellow]{final_embeddings.shape}[/bold yellow]")
    console.print(Panel.fit(resumen, border_style="green"))

if __name__ == "__main__":
    main()