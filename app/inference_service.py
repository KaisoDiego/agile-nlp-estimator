# app/inference_service.py
"""
Servicio de inferencia unificado: SBERT dual + PCA + LightGBM.

Responsabilidad única: dado un texto y los metadatos del Juez LLM,
devuelve la predicción continua y su contenedor Fibonacci.

Este módulo NO conoce FastAPI, Streamlit ni ningún framework de presentación.
Encapsula toda la lógica de ML para que los endpoints sean orquestadores delgados.

Pipeline de inferencia (alineado con Fase 3 + Fase 4 re-entrenadas):
  1. Embed user_story   → (1, 768) L2-normalizado  (P3-A)
  2. Embed defect_reasoning → (1, 768) L2-normalizado  (P3-B)
  3. hstack → (1, 1536)
  4. PCA.transform() → (1, 128)  (P4-D)
  5. Concatenar 7 features de calidad LLM  (P4-C)
  6. LightGBM.predict() con Huber loss  (P4-A)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.domain.fibonacci import clamp_minimum_sp, snap_to_fibonacci_ceiling

with open("config/settings.yaml", "r") as _f:
    _config = yaml.safe_load(_f)

_model_cfg = _config.get("model", {})
DOR_REJECTION_THRESHOLD: float = float(_model_cfg.get("dor_rejection_threshold", 3.0))
LONG_TAIL_SP_THRESHOLD: int = int(_model_cfg.get("long_tail_sp_threshold", 13))
LONG_TAIL_FRICTION_THRESHOLD: float = float(_model_cfg.get("long_tail_friction_threshold", 1.2))
LONG_TAIL_EXPONENT: float = float(_model_cfg.get("long_tail_exponent", 1.25))


@dataclass(frozen=True)
class InferenceResult:
    """Resultado completo de una inferencia sobre un PBI."""
    dor_estado: str
    q_score: float
    pbi_type: str
    defect_reasoning: str
    sp_raw_prediction: Optional[float]
    sp_adjusted_prediction: Optional[float]
    sp_fibonacci: Optional[int]
    model_version: str
    fibonacci_policy: str


class InferenceService:
    """
    Singleton de inferencia: carga SBERT y LightGBM una sola vez al arrancar.

    Uso en FastAPI:
        service = InferenceService()
        result  = service.predict(texto, hu_q_score, structural_score, semantic_score, ...)
    """

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer, self._sbert = self._load_sbert()
        self._lgbm, self._label_encoder, self._pca, self._metadata = self._load_lgbm()

    def _load_sbert(self) -> tuple:
        model_name = _config["vectorizer"]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sbert = AutoModel.from_pretrained(model_name).to(self._device)
        sbert.eval()
        return tokenizer, sbert

    def _load_lgbm(self) -> tuple:
        artefactos = joblib.load(_config["paths"]["model_save_path"])
        metadata = {
            "model_version": artefactos.get("split_strategy", "unknown"),
            "fibonacci_policy": artefactos.get("fibonacci_policy", "unknown"),
            "cv_metrics": artefactos.get("cv_metrics", {}),
        }
        return (
            artefactos["model"],
            artefactos["label_encoder"],
            artefactos["pca"],     # PCA ajustado en producción (P4-D)
            metadata,
        )

    @staticmethod
    def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9
        )

    def _embed(self, text: str) -> np.ndarray:
        """Devuelve vector (1, 768) L2-normalizado para un texto."""
        encoded = self._tokenizer(
            [text], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output = self._sbert(**encoded)
        pooled = self._mean_pooling(output, encoded["attention_mask"])
        normalized = F.normalize(pooled, p=2, dim=1)  # P3-A
        return normalized.cpu().numpy()

    def _encode_pbi_type(self, pbi_type: str) -> int:
        try:
            return int(self._label_encoder.transform([pbi_type])[0])
        except ValueError:
            return int(self._label_encoder.transform(["Technical Task"])[0])

    def _build_feature_vector(
        self,
        texto: str,
        hu_q_score: float,
        structural_score: float,
        semantic_score: float,
        pbi_type: str,
        defect_reasoning: str,
    ) -> np.ndarray:
        """
        Construye el vector de features completo alineado con el pipeline de entrenamiento.

        Returns:
            ndarray de forma (1, pca_components + 7) — listo para LightGBM.predict().
        """
        # P3-B: dual embedding
        emb_pbi = self._embed(texto)
        emb_reasoning = self._embed(defect_reasoning or "")
        emb_dual = np.hstack([emb_pbi, emb_reasoning])  # (1, 1536)

        # P4-D: reducción PCA
        emb_reduced = self._pca.transform(emb_dual)  # (1, 128)

        # P4-C: 7 features de calidad LLM
        pbi_encoded = self._encode_pbi_type(pbi_type)
        score_delta = structural_score - semantic_score
        gate_triggered = float(hu_q_score < DOR_REJECTION_THRESHOLD)
        score_product = structural_score * semantic_score

        quality = np.array([[
            hu_q_score,
            structural_score,
            semantic_score,
            score_delta,
            gate_triggered,
            score_product,
            pbi_encoded,
        ]])

        return np.hstack([emb_reduced, quality])  # (1, 135)

    def predict(
        self,
        texto: str,
        hu_q_score: float,
        structural_score: float,
        semantic_score: float,
        pbi_type: str,
        defect_reasoning: str,
        friction_multiplier: float = 1.0,
        is_long_tail_threshold: int = LONG_TAIL_SP_THRESHOLD,
        long_tail_friction_threshold: float = LONG_TAIL_FRICTION_THRESHOLD,
        long_tail_exponent: float = LONG_TAIL_EXPONENT,
    ) -> InferenceResult:
        """
        Pipeline completo sobre un ticket aprobado por el Guardián DoR.

        Args:
            texto: Texto completo del PBI.
            hu_q_score: Score compuesto del Juez LLM (1.0–5.0).
            structural_score: Completitud estructural G1-G5 (1.0–5.0).
            semantic_score: Claridad semántica del lenguaje (1.0–5.0).
            pbi_type: Clasificación taxonómica del Juez LLM.
            defect_reasoning: Justificación textual del Juez LLM.
            friction_multiplier: Factor de ajuste por capacidades del equipo.
        """
        if hu_q_score < DOR_REJECTION_THRESHOLD:
            return InferenceResult(
                dor_estado="Rechazado por DoR",
                q_score=hu_q_score,
                pbi_type=pbi_type,
                defect_reasoning=defect_reasoning,
                sp_raw_prediction=None,
                sp_adjusted_prediction=None,
                sp_fibonacci=None,
                model_version=self._metadata["model_version"],
                fibonacci_policy=self._metadata["fibonacci_policy"],
            )

        X = self._build_feature_vector(
            texto, hu_q_score, structural_score, semantic_score, pbi_type, defect_reasoning
        )

        raw = float(clamp_minimum_sp(self._lgbm.predict(X))[0])

        if (
            raw >= is_long_tail_threshold
            and friction_multiplier > long_tail_friction_threshold
        ):
            adjusted = raw * (friction_multiplier ** long_tail_exponent)
        else:
            adjusted = raw * friction_multiplier

        sp_fib = snap_to_fibonacci_ceiling(adjusted)

        return InferenceResult(
            dor_estado="Aprobado",
            q_score=hu_q_score,
            pbi_type=pbi_type,
            defect_reasoning=defect_reasoning,
            sp_raw_prediction=round(raw, 4),
            sp_adjusted_prediction=round(adjusted, 4),
            sp_fibonacci=sp_fib,
            model_version=self._metadata["model_version"],
            fibonacci_policy=self._metadata["fibonacci_policy"],
        )

    @property
    def model_metadata(self) -> dict:
        return self._metadata
