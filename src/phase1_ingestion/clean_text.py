# src/phase1_ingestion/clean_text.py
import re

def clean_pbi_text(text: str) -> str:
    """
    Limpia el texto del PBI eliminando caracteres problemáticos,
    pero preservando la jerga técnica, URLs y contexto.
    """
    if not isinstance(text, str):
        return ""
    
    # Reemplazar saltos de línea y retornos de carro por un espacio
    text = text.replace('\r', ' ').replace('\n', ' ')
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Quitar espacios al inicio y al final
    return text.strip()