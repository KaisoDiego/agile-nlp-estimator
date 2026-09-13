# src/phase4_prediction/group_cv.py
"""
Validación cruzada agrupada por project_id (GroupKFold).

Garantiza que ningún repositorio GitLab aparezca simultáneamente en entrenamiento
y prueba dentro de un fold, eliminando la fuga de vocabulario de dominio.

Mejoras implementadas:
- P4-A: Loss function Huber (alineada con MAE reportado en tesis).
- P4-B: Early stopping por fold con split interno 90/10.
- P4-D: PCA dentro de cada fold (fit solo en train) para balancear el espacio
        de features entre los 1536 dims SBERT y las 7 columnas de calidad LLM.
- P4-E: optimize_hyperparams() con Optuna (requiere: pip install optuna).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from src.domain.fibonacci import clamp_minimum_sp
from src.phase4_prediction.training_data import QUALITY_FEATURE_COLS, TrainingMatrix, build_features

GROUP_KFOLD_SPLITS = 5
RANDOM_STATE = 42
PCA_COMPONENTS = 128        # P4-D: SBERT 1536-dim → 128-dim dentro del fold
EARLY_STOPPING_ROUNDS = 30  # P4-B: paciencia para early stopping
INNER_VAL_FRACTION = 0.10   # P4-B: 10 % del train de cada fold como validación interna


@dataclass
class FoldMetrics:
    fold_index: int
    mae: float
    mdae: float
    r2: float
    n_train: int
    n_test: int
    best_iteration: int
    test_projects: list[str] = field(default_factory=list)


@dataclass
class GroupCVResult:
    fold_metrics: list[FoldMetrics]
    oof_predictions: np.ndarray
    oof_frame: pd.DataFrame

    @property
    def mae_mean(self) -> float:
        return float(np.mean([m.mae for m in self.fold_metrics]))

    @property
    def mae_std(self) -> float:
        return float(np.std([m.mae for m in self.fold_metrics]))

    @property
    def mae_weighted(self) -> float:
        """MAE ponderado por número de muestras por fold (estadísticamente correcto)."""
        total = sum(m.n_test for m in self.fold_metrics)
        return float(sum(m.mae * m.n_test / total for m in self.fold_metrics))

    @property
    def mdae_mean(self) -> float:
        return float(np.mean([m.mdae for m in self.fold_metrics]))

    @property
    def mdae_std(self) -> float:
        return float(np.std([m.mdae for m in self.fold_metrics]))


def create_lgbm_regressor(params: dict | None = None) -> lgb.LGBMRegressor:
    """
    Hiperparámetros base compartidos entre folds y modelo final de producción.

    P4-A: objective='huber' alinea la función de pérdida de entrenamiento con la
    métrica MAE reportada en la tesis, siendo más robusta ante la cola larga de
    story points (epics de 21-34 SP).

    Args:
        params: Dict opcional con hiperparámetros de Optuna que sobreescriben los base.
    """
    base: dict[str, Any] = {
        "objective": "huber",   # P4-A: MAE-aligned loss
        "alpha": 0.9,           # percentil del huber; 0.9 = conservador ante outliers
        "n_estimators": 500,    # más árboles + early stopping = busca el óptimo
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        base.update(params)
    return lgb.LGBMRegressor(**base)


def run_group_kfold_cv(
    matrix: TrainingMatrix,
    embeddings: np.ndarray,
    pca_components: int = PCA_COMPONENTS,
    lgbm_params: dict | None = None,
) -> GroupCVResult:
    """
    Ejecuta GroupKFold con PCA dentro del fold y early stopping (P4-B, P4-D).

    Por cada fold:
    1. Separa el train del fold en inner_train (90 %) e inner_val (10 %).
    2. Ajusta PCA solo sobre inner_train (sin leakage al test).
    3. Transforma inner_train, inner_val y test.
    4. Entrena LightGBM con eval_set=inner_val y early stopping.
    5. Predice sobre test con el mejor árbol encontrado.
    """
    gkf = GroupKFold(n_splits=GROUP_KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(matrix.target), dtype=float)
    fold_metrics: list[FoldMetrics] = []
    oof_frame = matrix.frame.copy()
    oof_frame["fold_id"] = -1

    n_samples = len(matrix.target)

    for fold_index, (train_idx, test_idx) in enumerate(
        gkf.split(np.zeros(n_samples), matrix.target, groups=matrix.groups)
    ):
        # P4-B: split interno para early stopping
        inner_val_size = max(1, int(INNER_VAL_FRACTION * len(train_idx)))
        inner_train_idx = train_idx[:-inner_val_size]
        inner_val_idx = train_idx[-inner_val_size:]

        # P4-D: PCA ajustado SOLO en inner_train (garantía de no-leakage)
        pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
        emb_inner_train = pca.fit_transform(embeddings[inner_train_idx])
        emb_inner_val = pca.transform(embeddings[inner_val_idx])
        emb_test = pca.transform(embeddings[test_idx])

        inner_train_matrix = TrainingMatrix(
            target=matrix.target[inner_train_idx],
            groups=matrix.groups[inner_train_idx],
            frame=matrix.frame.iloc[inner_train_idx].reset_index(drop=True),
        )
        inner_val_matrix = TrainingMatrix(
            target=matrix.target[inner_val_idx],
            groups=matrix.groups[inner_val_idx],
            frame=matrix.frame.iloc[inner_val_idx].reset_index(drop=True),
        )
        test_matrix = TrainingMatrix(
            target=matrix.target[test_idx],
            groups=matrix.groups[test_idx],
            frame=matrix.frame.iloc[test_idx].reset_index(drop=True),
        )

        X_inner_train, fold_encoder = build_features(
            inner_train_matrix, emb_inner_train, fit_encoder=True
        )
        X_inner_val, _ = build_features(
            inner_val_matrix, emb_inner_val, label_encoder=fold_encoder
        )
        X_test, _ = build_features(test_matrix, emb_test, label_encoder=fold_encoder)

        y_inner_train = matrix.target[inner_train_idx]
        y_inner_val = matrix.target[inner_val_idx]
        y_test = matrix.target[test_idx]

        # P4-B: early stopping con inner_val como señal de parada
        model = create_lgbm_regressor(lgbm_params)
        model.fit(
            X_inner_train,
            y_inner_train,
            eval_set=[(X_inner_val, y_inner_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        y_pred = clamp_minimum_sp(model.predict(X_test))
        oof_predictions[test_idx] = y_pred
        oof_frame.loc[matrix.frame.index[test_idx], "fold_id"] = fold_index

        test_projects = sorted(matrix.frame.iloc[test_idx]["project_id"].unique().tolist())

        fold_metrics.append(
            FoldMetrics(
                fold_index=fold_index,
                mae=float(mean_absolute_error(y_test, y_pred)),
                mdae=float(median_absolute_error(y_test, y_pred)),
                r2=float(r2_score(y_test, y_pred)),
                n_train=len(inner_train_idx),
                n_test=len(test_idx),
                best_iteration=int(model.best_iteration_),
                test_projects=test_projects,
            )
        )

    oof_frame["oof_pred_continuous"] = oof_predictions

    return GroupCVResult(
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        oof_frame=oof_frame,
    )


def fit_production_model(
    matrix: TrainingMatrix,
    embeddings: np.ndarray,
    pca_components: int = PCA_COMPONENTS,
    lgbm_params: dict | None = None,
) -> tuple[lgb.LGBMRegressor, LabelEncoder, PCA]:
    """
    Entrena el artefacto de producción sobre el 100 % de los datos curados.

    Las métricas reportadas en la tesis deben basarse en GroupKFold OOF, no en este fit.

    Returns:
        (model, label_encoder, pca) — los tres se guardan en el artefacto .pkl.
    """
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    emb_reduced = pca.fit_transform(embeddings)

    X, label_encoder = build_features(matrix, emb_reduced, fit_encoder=True)
    model = create_lgbm_regressor(lgbm_params)
    model.fit(X, matrix.target)
    return model, label_encoder, pca


# ──────────────────────────────────────────────────────────────────────────────
# P4-E: Búsqueda de hiperparámetros con Optuna
# ──────────────────────────────────────────────────────────────────────────────

def optimize_hyperparams(
    matrix: TrainingMatrix,
    embeddings: np.ndarray,
    n_trials: int = 40,
    pca_components: int = PCA_COMPONENTS,
    n_cv_splits: int = 3,
) -> dict:
    """
    Búsqueda bayesiana de hiperparámetros LightGBM con Optuna (P4-E).

    Usa 3 folds GroupKFold (en lugar de 5) con PCA dentro de cada fold
    para reducir el tiempo de búsqueda sin comprometer la validez estadística.

    Args:
        matrix: TrainingMatrix con todos los datos de entrenamiento.
        embeddings: Matriz de embeddings alineada con matrix.frame.
        n_trials: Número de evaluaciones Optuna (default=40, ~15 min).
        pca_components: Dimensiones PCA a usar durante la búsqueda.
        n_cv_splits: Folds para la búsqueda (default=3, menor que el CV final).

    Returns:
        Dict con los mejores hiperparámetros encontrados, listo para pasarse
        a create_lgbm_regressor() y run_group_kfold_cv().

    Raises:
        ImportError: Si optuna no está instalado (pip install optuna).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError(
            "Optuna no instalado. Ejecuta: pip install optuna"
        ) from exc

    gkf = GroupKFold(n_splits=n_cv_splits)
    n_samples = len(matrix.target)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "huber",
            "alpha": 0.9,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_maes: list[float] = []

        for train_idx, test_idx in gkf.split(
            np.zeros(n_samples), matrix.target, groups=matrix.groups
        ):
            pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
            emb_train = pca.fit_transform(embeddings[train_idx])
            emb_test = pca.transform(embeddings[test_idx])

            train_matrix = TrainingMatrix(
                target=matrix.target[train_idx],
                groups=matrix.groups[train_idx],
                frame=matrix.frame.iloc[train_idx].reset_index(drop=True),
            )
            test_matrix = TrainingMatrix(
                target=matrix.target[test_idx],
                groups=matrix.groups[test_idx],
                frame=matrix.frame.iloc[test_idx].reset_index(drop=True),
            )

            X_train, fold_encoder = build_features(train_matrix, emb_train, fit_encoder=True)
            X_test, _ = build_features(test_matrix, emb_test, label_encoder=fold_encoder)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, matrix.target[train_idx])
            y_pred = clamp_minimum_sp(model.predict(X_test))
            fold_maes.append(float(mean_absolute_error(matrix.target[test_idx], y_pred)))

        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params
