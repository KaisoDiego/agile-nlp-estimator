# src/phase4_prediction/training_data.py
"""
Construcción de la matriz de entrenamiento multimodal (SBERT + metadatos LLM).

Separa la lógica de I/O y filtrado del script de entrenamiento para mantener
train_lgbm.py como orquestador delgado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.domain.fibonacci import snap_to_fibonacci_ceiling

MAX_TARGET_SP = 40

# Columnas de calidad LLM expuestas al modelo (P4-C: añadidas score_delta,
# gate_triggered y score_product sobre las 4 originales).
QUALITY_FEATURE_COLS = [
    "hu_q_score",
    "structural_score",
    "semantic_score",
    "score_delta",       # structural - semantic: dirección del desequilibrio de calidad
    "gate_triggered",    # 1.0 si alguna compuerta G1-G5 activó el techo; 0.0 si no
    "score_product",     # structural × semantic: interacción multiplicativa
    "pbi_type_encoded",
]


@dataclass(frozen=True)
class TrainingMatrix:
    """Contenedor inmutable de target y metadatos de agrupación por proyecto."""

    target: np.ndarray
    groups: np.ndarray
    frame: pd.DataFrame


def load_scored_dataset(data_path: str, embeddings_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Carga el parquet puntuado (Fase 2) y la matriz de embeddings (Fase 3)."""
    df = pd.read_parquet(data_path)
    embeddings = np.load(embeddings_path)
    return df, embeddings


def filter_anomalies(
    df: pd.DataFrame, embeddings: np.ndarray
) -> tuple["TrainingMatrix", np.ndarray]:
    """
    Aplica filtros de calidad y retorna (TrainingMatrix, embeddings_filtrados).

    Filtros aplicados:
    - Épicos (target > 40 SP) descartados.
    - PBIs con evaluación LLM incompleta (pbi_type o hu_q_score = NaN)
      descartados — ocurren cuando la Fase 2 agota reintentos por rate-limit.

    Returns:
        Tupla (TrainingMatrix, np.ndarray) siempre alineados entre sí.

    Raises:
        ValueError: Si tras filtrar las filas no coinciden con los embeddings.
    """
    mask_sp = df["target"] <= MAX_TARGET_SP
    mask_llm = df["pbi_type"].notna() & df["hu_q_score"].notna()
    mask = mask_sp & mask_llm

    n_dropped_epics = (~mask_sp).sum()
    n_dropped_nan = (mask_sp & ~mask_llm).sum()
    if n_dropped_nan > 0:
        print(f"[filter_anomalies] {n_dropped_nan} PBIs descartados por evaluación LLM incompleta.")
    if n_dropped_epics > 0:
        print(f"[filter_anomalies] {n_dropped_epics} épicos descartados (target > {MAX_TARGET_SP} SP).")

    df_filtered = df.loc[mask].copy().reset_index(drop=True)
    embeddings_filtered = embeddings[mask.to_numpy()]

    if len(df_filtered) != embeddings_filtered.shape[0]:
        raise ValueError(
            f"Desalineación: {len(df_filtered)} filas vs {embeddings_filtered.shape[0]} embeddings."
        )

    if "project_id" not in df_filtered.columns:
        raise ValueError(
            "Columna 'project_id' requerida para GroupKFold. "
            "Ejecute Fase 1 de ingesta (ingest.py) antes del entrenamiento."
        )

    df_filtered["target_clean"] = df_filtered["target"].apply(snap_to_fibonacci_ceiling)
    groups = df_filtered["project_id"].to_numpy()

    matrix = TrainingMatrix(
        target=df_filtered["target_clean"].to_numpy(dtype=float),
        groups=groups,
        frame=df_filtered,
    )
    return matrix, embeddings_filtered


def build_features(
    matrix: TrainingMatrix,
    embeddings: np.ndarray,
    label_encoder: LabelEncoder | None = None,
    *,
    fit_encoder: bool = False,
) -> tuple[np.ndarray, LabelEncoder]:
    """
    Concatena embeddings SBERT con metadatos LLM extendidos (P4-C).

    Layout del vector de features (por fila):
        [SBERT_0 … SBERT_{D-1} | hu_q_score | structural_score | semantic_score |
         score_delta | gate_triggered | score_product | pbi_type_encoded]

        D = dimensión de embeddings tras PCA (ej. 128 si PCA_COMPONENTS=128)
        Total: D + 7 dimensiones

    Features derivadas (P4-C):
        score_delta    = structural_score - semantic_score
                         Captura la dirección del desequilibrio: PBIs con alta
                         estructura pero léxico vago vs. PBIs con lenguaje preciso
                         pero sin ACs.
        gate_triggered = float(hu_q_score < 3.0)
                         Flag binario: 1 indica que alguna compuerta G1-G5 activó
                         el techo. Señal directa de rechazo DoR.
        score_product  = structural_score × semantic_score
                         Interacción multiplicativa: penaliza fuertemente PBIs con
                         al menos una dimensión muy baja (ej. Struct:1 × Sem:4 = 4
                         vs Struct:4 × Sem:1 = 4 < Struct:3 × Sem:3 = 9).

    Args:
        matrix: Metadatos ya filtrados.
        embeddings: Matriz alineada con matrix.frame (puede ser post-PCA).
        label_encoder: Encoder existente (inferencia/CV). Si None, se instancia uno nuevo.
        fit_encoder: Si True, ajusta el encoder solo sobre matrix.frame (sin leakage cross-fold
                     cuando el caller pasa un subconjunto de entrenamiento).
    """
    df = matrix.frame.copy()
    encoder = label_encoder or LabelEncoder()

    if fit_encoder:
        df["pbi_type_encoded"] = encoder.fit_transform(df["pbi_type"])
    else:
        if label_encoder is None:
            raise ValueError("Se requiere label_encoder cuando fit_encoder=False.")
        df["pbi_type_encoded"] = encoder.transform(df["pbi_type"])

    # P4-C: features derivadas de los scores de calidad
    df["score_delta"] = df["structural_score"] - df["semantic_score"]
    df["gate_triggered"] = (df["hu_q_score"] < 3.0).astype(float)
    df["score_product"] = df["structural_score"] * df["semantic_score"]

    score_columns = df[QUALITY_FEATURE_COLS].to_numpy()
    features = np.hstack((embeddings, score_columns))

    print(
        f"[build_features] embeddings={embeddings.shape} | "
        f"score_cols={score_columns.shape} ({len(QUALITY_FEATURE_COLS)} features LLM) | "
        f"features_final={features.shape}"
        f"  → esperado (n, {embeddings.shape[1] + len(QUALITY_FEATURE_COLS)})"
    )

    return features, encoder
