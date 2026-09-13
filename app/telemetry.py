# app/telemetry.py
"""
Event Sourcing para el pipeline MLOps.

Cada interacción (predicción Jira o outcome GitLab) se persiste como un
evento JSON Lines (JSONL) en un archivo append-safe. Este log es la
fuente de verdad para:
  - Cruzar predicción <-> outcome por ticket_id.
  - Calcular medias de cohorte dinámicamente (sin diccionarios hardcodeados).
  - Detectar deriva del modelo (PSI, MAE rolling, bias direction).
  - Regenerar el dataset de reentrenamiento.

Schema de eventos:
  PredictionEvent  → se emite en /webhook/jira-ticket-created
  OutcomeEvent     → se emite en /webhook/gitlab-mr-merged
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

TELEMETRY_PATH = "data/mlops_events.jsonl"
SHOCK_MARGIN = 1.5        # Umbral de shock: lead_time > cohorte_mean * 1.5
MIN_COHORT_SAMPLES = 3    # Mínimo de muestras para calcular media dinámica de cohorte


# ──────────────────────────────────────────────────────────────
# SCHEMAS DE EVENTOS
# ──────────────────────────────────────────────────────────────

@dataclass
class PredictionEvent:
    """
    Emitido cuando Jira crea un ticket y el pipeline lo procesa.
    Vincula la predicción del modelo con el ticket_id para el cruce futuro.
    """
    event_type: str = field(default="prediction", init=False)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    ticket_id: str = ""
    texto_hash: str = ""          # SHA-256 del texto (privacidad + deduplicación)
    dor_estado: str = ""          # "Aprobado" | "Rechazado por DoR"
    q_score: float = 0.0
    pbi_type: str = ""
    dod_version: str = "unknown"

    # Sección modelo — null si rechazado por DoR
    model_version: str = ""
    fibonacci_policy: str = ""
    sp_raw_prediction: Optional[float] = None
    sp_fibonacci: Optional[int] = None

    # Sección fricción — null si el webhook no recibe perfil de equipo
    friction_multiplier: Optional[float] = None
    sp_fibonacci_adjusted: Optional[int] = None


@dataclass
class OutcomeEvent:
    """
    Emitido cuando GitLab fusiona un MR con el outcome real.
    Permite calcular residual = sp_original - sp_fibonacci del PredictionEvent correspondiente.
    """
    event_type: str = field(default="outcome", init=False)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    ticket_id: str = ""
    lead_time_days: float = 0.0
    archivos_modificados: int = 0
    sp_original: int = 0
    dod_version: str = "unknown"

    # Estadísticas de cohorte calculadas dinámicamente en el momento del evento
    lead_time_cohorte_mean: float = 0.0
    lead_time_cohorte_n: int = 0
    shock_threshold: float = 0.0
    shock_episodico: bool = False


# ──────────────────────────────────────────────────────────────
# FUNCIONES DE I/O
# ──────────────────────────────────────────────────────────────

def _append_event(event: PredictionEvent | OutcomeEvent) -> None:
    """Persiste un evento como línea JSON al archivo de telemetría (append-safe)."""
    os.makedirs(os.path.dirname(TELEMETRY_PATH), exist_ok=True)
    with open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def _load_outcome_events() -> pd.DataFrame:
    """Carga solo los eventos de tipo 'outcome' del log acumulado."""
    if not os.path.exists(TELEMETRY_PATH):
        return pd.DataFrame()

    rows = []
    with open(TELEMETRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("event_type") == "outcome":
                    rows.append(obj)
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# LÓGICA DE COHORTES (REEMPLAZA EL DICT HARDCODEADO)
# ──────────────────────────────────────────────────────────────

def _fallback_lead_time(sp: int) -> float:
    """
    Heurística de arranque (cold start): ratio empírico publicado en la literatura
    (aproximadamente 1.5 días de lead time por SP en equipos medianos).
    Solo se usa cuando no hay suficientes muestras históricas acumuladas.
    """
    return float(sp) * 1.5


def compute_cohort_lead_time(sp_original: int) -> tuple[float, int]:
    """
    Calcula la media de lead_time para la cohorte SP a partir de los eventos
    reales acumulados en mlops_events.jsonl.

    Returns:
        (media_dias, n_muestras) — si n < MIN_COHORT_SAMPLES, usa el fallback heurístico.
    """
    df = _load_outcome_events()

    if df.empty or "sp_original" not in df.columns:
        return _fallback_lead_time(sp_original), 0

    cohort = df[df["sp_original"] == sp_original]["lead_time_days"].dropna()
    n = len(cohort)

    if n < MIN_COHORT_SAMPLES:
        return _fallback_lead_time(sp_original), n

    return float(cohort.mean()), n


# ──────────────────────────────────────────────────────────────
# API PÚBLICA
# ──────────────────────────────────────────────────────────────

def hash_text(text: str) -> str:
    """SHA-256 del texto para identificar duplicados sin almacenar PII."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def emit_prediction_event(
    ticket_id: str,
    texto: str,
    dor_estado: str,
    q_score: float,
    pbi_type: str,
    dod_version: str,
    model_version: str,
    fibonacci_policy: str,
    sp_raw_prediction: Optional[float],
    sp_fibonacci: Optional[int],
    friction_multiplier: Optional[float] = None,
    sp_fibonacci_adjusted: Optional[int] = None,
) -> PredictionEvent:
    """Construye, persiste y devuelve el evento de predicción."""
    event = PredictionEvent(
        ticket_id=ticket_id,
        texto_hash=hash_text(texto),
        dor_estado=dor_estado,
        q_score=q_score,
        pbi_type=pbi_type,
        dod_version=dod_version,
        model_version=model_version,
        fibonacci_policy=fibonacci_policy,
        sp_raw_prediction=sp_raw_prediction,
        sp_fibonacci=sp_fibonacci,
        friction_multiplier=friction_multiplier,
        sp_fibonacci_adjusted=sp_fibonacci_adjusted,
    )
    _append_event(event)
    return event


def emit_outcome_event(
    ticket_id: str,
    lead_time_days: float,
    archivos_modificados: int,
    sp_original: int,
    dod_version: str,
) -> OutcomeEvent:
    """
    Calcula estadísticas de cohorte dinámicamente y persiste el evento de outcome.
    Nunca usa un diccionario estático de medias.
    """
    cohorte_mean, cohorte_n = compute_cohort_lead_time(sp_original)
    threshold = cohorte_mean * SHOCK_MARGIN
    shock = lead_time_days > threshold

    event = OutcomeEvent(
        ticket_id=ticket_id,
        lead_time_days=lead_time_days,
        archivos_modificados=archivos_modificados,
        sp_original=sp_original,
        dod_version=dod_version,
        lead_time_cohorte_mean=round(cohorte_mean, 4),
        lead_time_cohorte_n=cohorte_n,
        shock_threshold=round(threshold, 4),
        shock_episodico=shock,
    )
    _append_event(event)
    return event
