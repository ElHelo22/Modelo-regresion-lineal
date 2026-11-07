"""
analisis_ejemplo.py
Ejecuta el modelo de regresión de forma no interactiva y guarda un
reporte y los gráficos en la carpeta resultados/.

Uso:
    python app/analisis_ejemplo.py
    python app/analisis_ejemplo.py --csv datos/regresion.csv --y ventas_miles
    python app/analisis_ejemplo.py --csv mi_archivo.csv --y Y --x X1 X2 X3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(os.path.dirname(__file__))

from regresion import (  # noqa: E402
    ajustar_regresion,
    columnas_numericas,
    ecuacion_modelo,
    ejecutar_pruebas,
    leer_csv_desde_bytes,
)
from graficos_regresion import (  # noqa: E402
    grafico_correlacion,
    grafico_dispersion,
    grafico_qq,
    histograma_residuos,
    residuos_estandar_vs_ajustados,
)

RAIZ = Path(__file__).resolve().parent.parent
CSV_EJEMPLO = RAIZ / "datos" / "regresion.csv"
SALIDA_EJEMPLO = RAIZ / "resultados"


def construir_reporte(info_archivo: dict, df, y: str, xs: list[str], modelo: dict, pruebas: dict) -> str:
    lineas = [
        "=" * 72,
        "REPORTE DE REGRESIÓN LINEAL MULTIVARIADA",
        "Investigación Operativa II",
        "=" * 72,
        "",
        "1. DATOS",
        f"   Archivo: {info_archivo['nombre']}",
        f"   Delimitador: {info_archivo['delimitador']}",
        f"   Codificación: {info_archivo['codificacion']}",
        f"   Observaciones en el archivo: {len(df)}",
        f"   Variable dependiente: {y}",
        f"   Variables independientes: {', '.join(xs)}",
        "",
        "2. MODELO ESTIMADO",
        f"   {ecuacion_modelo(modelo)}",
        f"   Observaciones usadas: {modelo['n']}",
        f"   Filas descartadas: {modelo['filas_descartadas']}",
        f"   R²: {modelo['r2']:.4f}",
        f"   R² ajustado: {modelo['r2_ajustado']:.4f}",
        f"   Error estándar de la regresión: {modelo['error_estandar_regresion']:.4f}",
        f"   Estadístico F: {modelo['f']:.4f}  (p = {modelo['p_f']:.4f})",
        f"   Durbin-Watson: {modelo['durbin_watson']:.4f}",
        "",
        "3. COEFICIENTES",
        modelo["coeficientes"].to_string(
            index=False,
            float_format=lambda v: f"{v:,.4f}",
        ),
        "",
        "4. PRUEBAS DE VALIDACIÓN",
    ]

    normalidad = pruebas["normalidad"]
    if normalidad.get("aplicable"):
        lineas.extend([
            f"   Jarque-Bera: JB = {normalidad['jb']:.4f}, p = {normalidad['p_jb']:.4f}. "
            f"{normalidad['conclusion_jb']}",
            f"   Kolmogorov-Smirnov: D = {normalidad['ks']:.4f}, p = {normalidad['p_ks']:.4f}. "
            f"{normalidad['conclusion_ks']}",
        ])
    else:
        lineas.append(f"   Normalidad: {normalidad.get('mensaje', 'no aplicable')}")

    dw = pruebas["durbin_watson"]
    lineas.append(
        f"   Durbin-Watson: {dw['estadistico']:.4f}. {dw['conclusion']}"
    )

    bp = pruebas["breusch_pagan"]
    if bp.get("aplicable"):
        lineas.append(
            f"   Breusch-Pagan: LM = {bp['estadistico']:.4f}, gl = {bp['gl']}, "
            f"p = {bp['p']:.4f}. {bp['conclusion']}"
        )
    else:
        lineas.append(f"   Breusch-Pagan: {bp.get('mensaje', 'no aplicable')}")

    white = pruebas["white"]
    if white.get("aplicable"):
        lineas.append(
            f"   White: LM = {white['estadistico']:.4f}, gl = {white['gl']}, "
            f"p = {white['p']:.4f}. {white['conclusion']}"
        )
    else:
        lineas.append(f"   White: {white.get('mensaje', 'no aplicable')}")

    lineas.extend([
        "",
        "5. MULTICOLINEALIDAD (VIF)",
        pruebas["vif"].to_string(float_format=lambda v: f"{v:,.4f}"),
        "",
        "=" * 72,
    ])
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estima un modelo de regresión lineal multivariada y exporta resultados."
    )
    parser.add_argument(
        "--csv",
        default=str(CSV_EJEMPLO),
        help="Ruta del archivo CSV. Por defecto usa datos/regresion.csv.",
    )
    parser.add_argument(
        "--y",
        default=None,
        help="Variable dependiente. Por defecto usa la primera columna numérica.",
    )
    parser.add_argument(
        "--x",
        nargs="+",
        default=None,
        help="Variables independientes. Por defecto usa el resto de columnas numéricas.",
    )
    parser.add_argument(
        "--salida",
        default=str(SALIDA_EJEMPLO),
        help="Carpeta donde guardar el reporte y los gráficos.",
    )
    args = parser.parse_args()

    ruta_csv = Path(args.csv)
    if not ruta_csv.exists():
        raise SystemExit(f"No existe el archivo: {ruta_csv}")

    df, delimitador, codificacion = leer_csv_desde_bytes(ruta_csv.read_bytes())
    numericas = columnas_numericas(df)
    if not numericas:
        raise SystemExit("El archivo no contiene variables numéricas.")

    y = args.y or numericas[0]
    xs = args.x or [c for c in numericas if c != y]
    if not xs:
        raise SystemExit("Se necesita al menos una variable independiente.")

    modelo = ajustar_regresion(df, y, xs)
    pruebas = ejecutar_pruebas(modelo)

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    info = {
        "nombre": ruta_csv.name,
        "delimitador": delimitador,
        "codificacion": codificacion,
    }
    reporte = construir_reporte(info, df, y, xs, modelo, pruebas)
    (salida / "reporte_regresion.txt").write_text(reporte, encoding="utf-8")

    graficos = {
        "histograma_residuos.png": histograma_residuos(modelo["residuos"]),
        "qq_residuos.png": grafico_qq(modelo["residuos"]),
        "residuos_estandarizados.png": residuos_estandar_vs_ajustados(
            modelo["ajustados"], modelo["residuos_estandarizados"]
        ),
        "matriz_correlaciones.png": grafico_correlacion(df, [y, *xs]),
    }
    for nombre, figura in graficos.items():
        figura.savefig(salida / nombre, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(figura)

    for indice, variable in enumerate(xs, start=1):
        figura = grafico_dispersion(df, y, variable)
        figura.savefig(
            salida / f"dispersion_{indice}_{variable}.png",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figura)

    print("Análisis finalizado correctamente.")
    print(f"Modelo: {ecuacion_modelo(modelo)}")
    print(f"R² = {modelo['r2']:.4f} | R² ajustado = {modelo['r2_ajustado']:.4f}")
    print(f"Resultados guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
