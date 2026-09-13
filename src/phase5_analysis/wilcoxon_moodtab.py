"""
Prueba de Wilcoxon signed-rank (pareada) sobre el experimento MOODTAB.

Hipótesis:
  H0: La mediana de las diferencias |Error_Equipo| - |Error_IA| = 0
  H1: Las distribuciones de error difieren significativamente (bilateral)

Referencia: Wilcoxon (1945); scipy.stats.wilcoxon usa el método exacto para N<=25.
"""

import sys
import io

# Forzar UTF-8 en la terminal de Windows (cp1252 no soporta unicode matemático)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from scipy import stats
from rich.console import Console
from rich.table import Table

CSV_PATH = "reports/auditoria_forense_moodtab.csv"

console = Console()


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    aprobados = df[df["DoR_Estado"] == "Aprobado"].copy()
    aprobados = aprobados.dropna(subset=["Error_Horas_Equipo", "Error_Horas_IA"])

    n = len(aprobados)
    console.print(f"\n[bold cyan]Wilcoxon signed-rank — MOODTAB (N={n} tickets aprobados)[/bold cyan]\n")

    err_equipo = aprobados["Error_Horas_Equipo"].values.astype(float)
    err_ia = aprobados["Error_Horas_IA"].values.astype(float)
    diferencias = err_equipo - err_ia

    tabla = Table(title="Errores absolutos por ticket (horas)")
    tabla.add_column("Ticket", style="dim")
    tabla.add_column("Título", max_width=28)
    tabla.add_column("Err Equipo (h)", justify="right")
    tabla.add_column("Err IA (h)", justify="right")
    tabla.add_column("Diferencia", justify="right")
    tabla.add_column("Ganador", justify="center")

    for _, row in aprobados.iterrows():
        diff = row["Error_Horas_Equipo"] - row["Error_Horas_IA"]
        color = "green" if diff > 0 else ("red" if diff < 0 else "yellow")
        tabla.add_row(
            str(row["Ticket_ID"]),
            str(row["Titulo"])[:28],
            f"{row['Error_Horas_Equipo']:.1f}",
            f"{row['Error_Horas_IA']:.1f}",
            f"[{color}]{diff:+.1f}[/{color}]",
            str(row["Ganador_Estimacion"]),
        )

    console.print(tabla)

    # ---------- estadísticos descriptivos ----------
    mae_equipo = np.mean(err_equipo)
    mae_ia = np.mean(err_ia)
    medae_equipo = np.median(err_equipo)
    medae_ia = np.median(err_ia)

    desc = Table(title="Estadísticos descriptivos")
    desc.add_column("Métrica", style="cyan")
    desc.add_column("Equipo Humano", justify="right")
    desc.add_column("Modelo IA", justify="right")
    desc.add_row("MAE (h)", f"{mae_equipo:.2f}", f"{mae_ia:.2f}")
    desc.add_row("MdAE (h)", f"{medae_equipo:.2f}", f"{medae_ia:.2f}")
    desc.add_row("Diferencia media (Eq - IA)", f"{np.mean(diferencias):+.2f}", "")
    desc.add_row("Desv. estándar diferencias", f"{np.std(diferencias, ddof=1):.2f}", "")
    console.print(desc)

    # ---------- Wilcoxon signed-rank ----------
    # alternative='two-sided': H1 bilateral (alguno es mejor)
    # method='exact' es el default para N<=25 en scipy >= 1.7
    stat, p_value = stats.wilcoxon(err_equipo, err_ia, alternative="two-sided")

    # Tamaño del efecto r = Z / sqrt(N)
    # scipy no expone Z directamente; lo recalculamos desde la distribución normal
    # para N suficientemente pequeño usamos la aproximación z basada en la stat T
    n_pairs = len(diferencias[diferencias != 0])
    mu_T = n_pairs * (n_pairs + 1) / 4
    sigma_T = np.sqrt(n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24)
    z_approx = (stat - mu_T) / sigma_T if sigma_T > 0 else float("nan")
    effect_r = abs(z_approx) / np.sqrt(n)

    resultado = Table(title="Resultado del test de Wilcoxon (bilateral, alpha=0.05)")
    resultado.add_column("Estadístico", style="cyan")
    resultado.add_column("Valor", justify="right")
    resultado.add_row("W (suma rangos menores)", f"{stat:.4f}")
    resultado.add_row("p-valor", f"{p_value:.4f}")
    resultado.add_row("z aproximado", f"{z_approx:.4f}")
    resultado.add_row("Tamaño del efecto r", f"{effect_r:.4f}")
    resultado.add_row("Pares no empatados (N')", str(n_pairs))
    console.print(resultado)

    # ---------- interpretación ----------
    alpha = 0.05
    console.print()
    if p_value < alpha:
        console.print(
            f"[bold green]→ p={p_value:.4f} < {alpha}: Se rechaza H0.[/bold green] "
            "Existe diferencia significativa entre los errores del equipo y la IA."
        )
    else:
        console.print(
            f"[bold yellow]→ p={p_value:.4f} ≥ {alpha}: No se rechaza H0.[/bold yellow] "
            f"Con N={n} tickets no hay evidencia estadística suficiente para afirmar "
            "que un estimador supera al otro (consistente con lo declarado en §9.4.2)."
        )

    umbral_r = {"pequeño": 0.1, "mediano": 0.3, "grande": 0.5}
    if effect_r < umbral_r["pequeño"]:
        etiqueta_r = "despreciable"
    elif effect_r < umbral_r["mediano"]:
        etiqueta_r = "pequeño"
    elif effect_r < umbral_r["grande"]:
        etiqueta_r = "mediano"
    else:
        etiqueta_r = "grande"

    console.print(
        f"   Tamaño del efecto r={effect_r:.3f} → [{etiqueta_r}] "
        "(Cohen: pequeño≥0.1, mediano≥0.3, grande≥0.5)"
    )

    # ---------- nota de potencia estadística ----------
    console.print()
    console.print(
        "[dim]Nota: El test de Wilcoxon requiere ~20-25 pares para detectar efectos medianos "
        f"(β=0.80, α=0.05). Con N={n} el poder es insuficiente para cualquier conclusión "
        "definitiva; este resultado es consistente con la limitación declarada en §11.2.[/dim]"
    )


if __name__ == "__main__":
    main()
