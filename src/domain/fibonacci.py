# src/domain/fibonacci.py
"""
Reglas de cuantización Fibonacci compartidas por entrenamiento e inferencia.

Política única (conservadora): mapear un valor continuo al contenedor Fibonacci
igual o inmediatamente superior (techo). Evita subestimar esfuerzo en la escala ágil.
"""

from __future__ import annotations

import numpy as np

# Secuencia estándar usada en la tesis (contenedores discretos de Planning Poker).
FIBONACCI_SP_SEQUENCE: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34, 55)


def snap_to_fibonacci_ceiling(value: float) -> int:
    """
    Redondea hacia el contenedor Fibonacci >= value (techo conservador).

    Args:
        value: Esfuerzo continuo o entero observado.

    Returns:
        Entero en la secuencia Fibonacci. Si value supera el máximo, devuelve 55.
    """
    if value <= 0:
        return FIBONACCI_SP_SEQUENCE[0]

    for fib in FIBONACCI_SP_SEQUENCE:
        if value <= fib:
            return fib

    return FIBONACCI_SP_SEQUENCE[-1]


def snap_array_to_fibonacci_ceiling(values: np.ndarray) -> np.ndarray:
    """Aplica snap_to_fibonacci_ceiling de forma vectorizada sobre un array 1-D."""
    vectorized = np.vectorize(snap_to_fibonacci_ceiling, otypes=[int])
    return vectorized(values)


def clamp_minimum_sp(values: np.ndarray, minimum: float = 1.0) -> np.ndarray:
    """
    Impone piso mínimo en predicciones continuas antes de cuantizar.

    En Fibonacci ágil no existe '0 SP' para PBIs aceptados en refinamiento.
    """
    return np.maximum(values, minimum)
