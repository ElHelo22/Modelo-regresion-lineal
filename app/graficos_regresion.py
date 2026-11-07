"""
graficos_regresion.py
Gráficos del Sistema de Regresión Lineal Multivariada:
dispersión Xi vs Y, histograma de residuos, Q-Q y residuos
estandarizados contra valores ajustados.
"""

from __future__ import annotations

import io
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_PRINCIPAL = "#1f5f8b"
COLOR_ACENTO = "#e07a1f"
COLOR_ALERTA = "#c0392b"


def _estilo(ax):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_facecolor("#fbfbfb")
    for borde in ("top", "right"):
        ax.spines[borde].set_visible(False)


def _nueva_figura(ancho=7.0, alto=4.5):
    fig, ax = plt.subplots(figsize=(ancho, alto), constrained_layout=True)
    _estilo(ax)
    return fig, ax


def grafico_dispersion(df: pd.DataFrame, dependiente: str, independiente: str):
    """Diagrama de dispersión de una variable Xi frente a Y, con recta de tendencia."""
    fig, ax = _nueva_figura()
    x = pd.to_numeric(df[independiente], errors="coerce")
    y = pd.to_numeric(df[dependiente], errors="coerce")
    validos = pd.DataFrame({independiente: x, dependiente: y}).dropna()

    ax.scatter(
        validos[independiente], validos[dependiente],
        alpha=0.72, color=COLOR_PRINCIPAL, edgecolor="white", linewidth=0.4,
        label="Observaciones",
    )
    if len(validos) >= 2 and validos[independiente].nunique() > 1:
        pendiente, intercepto = np.polyfit(
            validos[independiente].to_numpy(dtype=float),
            validos[dependiente].to_numpy(dtype=float),
            deg=1,
        )
        xs = np.linspace(validos[independiente].min(), validos[independiente].max(), 100)
        ax.plot(xs, intercepto + pendiente * xs, color=COLOR_ACENTO, linewidth=2.0,
                label="Recta de tendencia")
        correlacion = validos[independiente].corr(validos[dependiente])
        ax.text(
            0.02, 0.98, f"r = {correlacion:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85,
                  "edgecolor": "#cccccc"},
        )

    ax.set_title(f"{dependiente} vs. {independiente}", fontsize=12, fontweight="bold")
    ax.set_xlabel(independiente)
    ax.set_ylabel(dependiente)
    ax.legend(frameon=False, loc="lower right")
    return fig


def histograma_residuos(residuos: np.ndarray):
    """Histograma de residuos con curva normal de referencia."""
    fig, ax = _nueva_figura()
    residuos = np.asarray(residuos, dtype=float)
    residuos = residuos[np.isfinite(residuos)]

    ax.hist(residuos, bins="auto", density=True, color=COLOR_PRINCIPAL,
            alpha=0.62, edgecolor="white", linewidth=0.7, label="Residuos")

    if len(residuos) > 1 and np.std(residuos, ddof=1) > 0:
        media = float(np.mean(residuos))
        desv = float(np.std(residuos, ddof=1))
        xs = np.linspace(residuos.min(), residuos.max(), 300)
        normal = NormalDist(media, desv)
        ys = np.array([normal.pdf(float(v)) for v in xs])
        ax.plot(xs, ys, color=COLOR_ACENTO, linewidth=2.0,
                label=f"Normal (μ={media:.2f}, σ={desv:.2f})")

    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("Histograma de residuos", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Densidad")
    ax.legend(frameon=False)
    return fig


def grafico_qq(residuos: np.ndarray):
    """Gráfico Q-Q de los residuos contra la distribución normal."""
    fig, ax = _nueva_figura()
    residuos = np.asarray(residuos, dtype=float)
    residuos = residuos[np.isfinite(residuos)]
    n = len(residuos)

    if n < 2:
        ax.text(0.5, 0.5, "Se necesitan al menos 2 residuos",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    media = float(np.mean(residuos))
    desv = float(np.std(residuos, ddof=1))
    if desv <= 0:
        ax.text(0.5, 0.5, "Los residuos no tienen variabilidad",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    estandarizados = (residuos - media) / desv
    ordenados = np.sort(estandarizados)
    probabilidades = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    normal = NormalDist()
    cuantiles_teoricos = np.array([normal.inv_cdf(float(p)) for p in probabilidades])

    ax.scatter(cuantiles_teoricos, ordenados, color=COLOR_PRINCIPAL, alpha=0.78,
               edgecolor="white", linewidth=0.4, label="Cuantiles observados")
    minimo = min(cuantiles_teoricos.min(), ordenados.min())
    maximo = max(cuantiles_teoricos.max(), ordenados.max())
    ax.plot([minimo, maximo], [minimo, maximo], color=COLOR_ACENTO,
            linewidth=1.8, linestyle="--", label="Referencia normal")

    ax.set_title("Gráfico Q-Q normal de residuos", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cuantiles teóricos normales")
    ax.set_ylabel("Residuos estandarizados observados")
    ax.legend(frameon=False, loc="upper left")
    return fig


def residuos_estandar_vs_ajustados(ajustados: np.ndarray, residuos_estandarizados: np.ndarray):
    """Gráfico de residuos estandarizados contra valores ajustados."""
    fig, ax = _nueva_figura()
    ajustados = np.asarray(ajustados, dtype=float)
    residuos = np.asarray(residuos_estandarizados, dtype=float)
    validos = np.isfinite(ajustados) & np.isfinite(residuos)

    ax.scatter(ajustados[validos], residuos[validos], alpha=0.72,
               color=COLOR_PRINCIPAL, edgecolor="white", linewidth=0.4,
               label="Residuos estandarizados")
    ax.axhline(0.0, color="#444444", linewidth=1.2)
    for nivel in (-2.0, 2.0):
        ax.axhline(nivel, color=COLOR_ALERTA, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title("Residuos estandarizados vs. valores ajustados",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Valor ajustado")
    ax.set_ylabel("Residuo estandarizado")
    ax.legend(frameon=False, loc="upper right")
    return fig


def grafico_correlacion(df: pd.DataFrame, variables: list[str]):
    """Mapa de calor de correlaciones entre las variables del modelo."""
    datos = df[variables].apply(pd.to_numeric, errors="coerce")
    correlaciones = datos.corr()
    fig, ax = plt.subplots(
        figsize=(max(5.5, 1.1 * len(variables) + 2.5), max(4.2, 0.9 * len(variables) + 2.0)),
        constrained_layout=True,
    )
    imagen = ax.imshow(correlaciones, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(variables)))
    ax.set_yticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(variables, fontsize=8)
    for i in range(len(variables)):
        for j in range(len(variables)):
            valor = correlaciones.iloc[i, j]
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(valor) > 0.55 else "#222222")
    ax.set_title("Matriz de correlaciones", fontsize=12, fontweight="bold")
    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    return fig


def figura_a_png(fig) -> bytes:
    """Convierte una figura Matplotlib a bytes PNG para descarga."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight",
                facecolor="white")
    buffer.seek(0)
    return buffer.getvalue()
