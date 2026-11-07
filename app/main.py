"""
main.py
Sistema de Regresión Lineal Multivariada
Interfaz web interactiva construida con Streamlit.

Ejecutar con:  streamlit run app/main.py

Práctica de Investigación Operativa II
Docente: M.Sc. Ing. Juan Carlos Catunta Choquecalle
Gestión 2026
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from regresion import (  # noqa: E402
    ajustar_regresion,
    calcular_vif,
    columnas_numericas,
    detectar_atipicos_iqr,
    ecuacion_modelo,
    estadisticas_descriptivas,
    ejecutar_pruebas,
    leer_csv_desde_bytes,
    limpiar_datos,
)
from graficos_regresion import (  # noqa: E402
    figura_a_png,
    grafico_correlacion,
    grafico_dispersion,
    grafico_qq,
    histograma_residuos,
    residuos_estandar_vs_ajustados,
)

RUTA_DATOS = os.path.join(os.path.dirname(__file__), "..", "datos")
RUTA_EJEMPLO = os.path.join(RUTA_DATOS, "regresion.csv")

st.set_page_config(
    page_title="Regresión Lineal Multivariada",
    page_icon="📊",
    layout="wide",
)


# ----------------------------------------------------------------------
# Utilidades de estado y formato
# ----------------------------------------------------------------------
def cargar_datos_ejemplo() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RUTA_EJEMPLO, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df, {
        "nombre": "regresion.csv (datos de ejemplo)",
        "delimitador": ",",
        "codificacion": "utf-8",
        "origen": "ejemplo",
    }


def reiniciar_modelo() -> None:
    st.session_state.modelo = None
    st.session_state.pruebas = None
    st.session_state.reporte_texto = None


def inicializar_estado() -> None:
    if "df_crudo" not in st.session_state:
        try:
            df, info = cargar_datos_ejemplo()
            st.session_state.df_crudo = df
            st.session_state.info_carga = info
        except Exception as error:  # pragma: no cover - depende del archivo local
            st.session_state.df_crudo = None
            st.session_state.info_carga = {"error": str(error)}
        st.session_state.df_limpio = None
        st.session_state.reporte_limpieza = None
        st.session_state.modelo = None
        st.session_state.pruebas = None
        st.session_state.reporte_texto = None
        st.session_state.ultimo_archivo = None

    for clave, valor in {
        "df_limpio": None,
        "reporte_limpieza": None,
        "modelo": None,
        "pruebas": None,
        "reporte_texto": None,
        "ultimo_archivo": None,
    }.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def numero(valor, decimales: int = 4) -> str:
    if valor is None:
        return "—"
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if np.isnan(valor):
        return "—"
    if np.isinf(valor):
        return "∞"
    return f"{valor:,.{decimales}f}"


def datos_trabajo() -> pd.DataFrame | None:
    if st.session_state.df_limpio is not None:
        return st.session_state.df_limpio
    return st.session_state.df_crudo


def mostrar_advertencia_seleccion(modelo: dict, y: str | None, xs: list[str]) -> None:
    if y != modelo["dependiente"] or xs != modelo["independientes"]:
        st.warning(
            "La selección actual de variables no coincide con el modelo estimado. "
            "El modelo, las pruebas y los gráficos mostrados corresponden a "
            f"Y = {modelo['dependiente']} y X = {', '.join(modelo['independientes'])}. "
            "Vuelva a la sección de estimación para ajustar el modelo con la nueva selección."
        )


def tabla_coeficientes(modelo: dict):
    tabla = modelo["coeficientes"].copy()
    return tabla.style.format({
        "Coeficiente": "{:,.4f}",
        "Error estándar": "{:,.4f}",
        "Estadístico t": "{:,.4f}",
        "p-valor": "{:,.4f}",
    })


def tabla_vif(vif: pd.DataFrame):
    def resaltar(fila):
        valor = fila["VIF"]
        alerta = np.isinf(valor) or (np.isfinite(valor) and valor > 10)
        return ["background-color: #ffd9d9; color: #8b1a1a" if alerta else ""] * len(fila)

    return vif.style.apply(resaltar, axis=1).format({
        "R² auxiliar": "{:,.4f}",
        "VIF": "{:,.3f}",
    })


def generar_reporte_texto(
    info_carga: dict,
    df: pd.DataFrame,
    y: str,
    xs: list[str],
    modelo: dict,
    pruebas: dict,
) -> str:
    lineas = []
    lineas.append("=" * 72)
    lineas.append("REPORTE DE REGRESIÓN LINEAL MULTIVARIADA")
    lineas.append("Investigación Operativa II")
    lineas.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lineas.append("=" * 72)
    lineas.append("")
    lineas.append("1. DATOS")
    lineas.append(f"   Archivo: {info_carga.get('nombre', '—')}")
    lineas.append(f"   Delimitador: {info_carga.get('delimitador', '—')}")
    lineas.append(f"   Observaciones disponibles: {len(df)}")
    lineas.append(f"   Variable dependiente: {y}")
    lineas.append(f"   Variables independientes: {', '.join(xs)}")
    reporte_limpieza = st.session_state.get("reporte_limpieza")
    if reporte_limpieza:
        lineas.append(
            "   Limpieza: "
            f"{reporte_limpieza['filas_iniciales']} filas iniciales, "
            f"{reporte_limpieza['filas_finales']} filas finales, "
            f"{reporte_limpieza['faltantes_finales']} faltantes restantes."
        )
    lineas.append("")
    lineas.append("2. MODELO ESTIMADO")
    lineas.append(f"   {ecuacion_modelo(modelo)}")
    lineas.append(f"   Observaciones usadas: {modelo['n']}")
    lineas.append(f"   Filas descartadas por datos faltantes: {modelo['filas_descartadas']}")
    lineas.append(f"   R²: {modelo['r2']:.4f}")
    lineas.append(f"   R² ajustado: {modelo['r2_ajustado']:.4f}")
    lineas.append(f"   Error estándar de la regresión: {modelo['error_estandar_regresion']:.4f}")
    lineas.append(f"   Estadístico F: {modelo['f']:.4f}  (p = {modelo['p_f']:.4f})")
    lineas.append(f"   Durbin-Watson: {modelo['durbin_watson']:.4f}")
    lineas.append("")
    lineas.append("3. COEFICIENTES")
    tabla = modelo["coeficientes"].copy()
    lineas.append(
        tabla.to_string(
            index=False,
            float_format=lambda v: f"{v:,.4f}",
        )
    )
    lineas.append("")
    lineas.append("4. PRUEBAS DE VALIDACIÓN")
    normalidad = pruebas["normalidad"]
    if normalidad.get("aplicable"):
        lineas.append(
            f"   Normalidad Jarque-Bera: JB = {normalidad['jb']:.4f}, "
            f"p = {normalidad['p_jb']:.4f}. {normalidad['conclusion_jb']}"
        )
        lineas.append(
            f"   Normalidad Kolmogorov-Smirnov: D = {normalidad['ks']:.4f}, "
            f"p = {normalidad['p_ks']:.4f}. {normalidad['conclusion_ks']}"
        )
    else:
        lineas.append(f"   Normalidad: {normalidad.get('mensaje', 'no aplicable')}")

    dw = pruebas["durbin_watson"]
    lineas.append(f"   Autocorrelación Durbin-Watson: {dw['estadistico']:.4f}. {dw['conclusion']}")

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

    lineas.append("")
    lineas.append("5. MULTICOLINEALIDAD (VIF)")
    lineas.append(
        pruebas["vif"].to_string(float_format=lambda v: f"{v:,.4f}")
    )
    lineas.append("")
    lineas.append("=" * 72)
    return "\n".join(lineas)


# ----------------------------------------------------------------------
# Inicialización
# ----------------------------------------------------------------------
inicializar_estado()

st.title("📊 Sistema de Regresión Lineal Multivariada")
st.caption(
    "Carga, limpieza, estimación MCO, validación estadística y gráficos — "
    "Investigación Operativa II"
)

df_base = datos_trabajo()

# ----------------------------------------------------------------------
# Barra lateral: menú y selección de variables
# ----------------------------------------------------------------------
st.sidebar.title("Navegación")
menu = st.sidebar.radio(
    "Sección",
    [
        "1. Carga y limpieza",
        "2. Estadísticas descriptivas",
        "3. Estimación del modelo",
        "4. Validación del modelo",
        "5. Gráficos",
        "6. Reporte final",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Variables del modelo")

if df_base is not None and columnas_numericas(df_base):
    numericas = columnas_numericas(df_base)
    if "var_y" not in st.session_state or st.session_state.var_y not in numericas:
        st.session_state.var_y = numericas[0]

    st.sidebar.selectbox(
        "Variable dependiente (Y)",
        numericas,
        key="var_y",
    )

    opciones_x = [c for c in numericas if c != st.session_state.var_y]
    if "var_x" not in st.session_state:
        st.session_state.var_x = opciones_x[:]
    else:
        st.session_state.var_x = [c for c in st.session_state.var_x if c in opciones_x]

    st.sidebar.multiselect(
        "Variables independientes (X)",
        opciones_x,
        key="var_x",
        help="Puede seleccionar una o varias variables explicativas numéricas.",
    )
    y_sel = st.session_state.var_y
    xs_sel = list(st.session_state.var_x)
    st.sidebar.caption(
        f"Filas disponibles: **{len(df_base)}** · "
        f"Variables numéricas: **{len(numericas)}**"
    )
else:
    y_sel, xs_sel = None, []
    st.sidebar.info("Cargue un archivo CSV con variables numéricas.")

st.sidebar.divider()
if st.session_state.modelo is not None:
    st.sidebar.success(
        "Modelo estimado: "
        f"Y = {st.session_state.modelo['dependiente']}"
    )
else:
    st.sidebar.caption("Aún no hay un modelo estimado.")


# ----------------------------------------------------------------------
# 1. Carga y limpieza
# ----------------------------------------------------------------------
if menu.startswith("1"):
    st.header("1. Carga y limpieza de datos")
    st.write(
        "Cargue un archivo CSV con la variable dependiente y las variables "
        "independientes. El sistema detecta automáticamente el delimitador, "
        "muestra una vista previa y permite tratar valores faltantes y atípicos."
    )

    archivo = st.file_uploader(
        "Seleccione un archivo CSV",
        type=["csv"],
        help="Formatos admitidos: separado por coma, punto y coma, tabulación o barra vertical.",
    )

    if archivo is not None:
        identificador = f"{archivo.name}-{len(archivo.getvalue())}"
        if st.session_state.ultimo_archivo != identificador:
            try:
                df_nuevo, delimitador, codificacion = leer_csv_desde_bytes(archivo.getvalue())
            except Exception as error:
                st.error(f"No se pudo leer el archivo: {error}")
            else:
                st.session_state.df_crudo = df_nuevo
                st.session_state.info_carga = {
                    "nombre": archivo.name,
                    "delimitador": delimitador,
                    "codificacion": codificacion,
                    "origen": "usuario",
                }
                st.session_state.df_limpio = None
                st.session_state.reporte_limpieza = None
                st.session_state.ultimo_archivo = identificador
                reiniciar_modelo()
                st.rerun()

    df_crudo = st.session_state.df_crudo
    info = st.session_state.info_carga or {}

    if df_crudo is None:
        st.error(
            "No hay datos cargados. Suba un archivo CSV para comenzar "
            "o verifique que exista datos/regresion.csv."
        )
    else:
        st.subheader("Información del archivo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Archivo", info.get("nombre", "—"))
        c2.metric("Filas", len(df_crudo))
        c3.metric("Columnas", len(df_crudo.columns))
        c4.metric("Delimitador", repr(info.get("delimitador", "—")))

        if info.get("codificacion"):
            st.caption(f"Codificación detectada: {info['codificacion']}")

        st.subheader("Vista previa de las primeras filas")
        st.dataframe(df_crudo.head(10), width="stretch")

        with st.expander("Tipos de datos y valores faltantes"):
            resumen_tipos = pd.DataFrame({
                "Tipo": [str(t) for t in df_crudo.dtypes],
                "No nulos": df_crudo.notna().sum().to_numpy(),
                "Faltantes": df_crudo.isna().sum().to_numpy(),
                "Porcentaje faltante": (
                    100.0 * df_crudo.isna().mean()
                ).round(2).to_numpy(),
            }, index=df_crudo.columns)
            st.dataframe(resumen_tipos, width="stretch")

        st.subheader("Limpieza de datos")
        with st.form("formulario_limpieza"):
            estrategia = st.selectbox(
                "Tratamiento de valores faltantes",
                [
                    "Eliminar filas con faltantes",
                    "Imputar con la media",
                    "Imputar con la mediana",
                    "No imputar",
                ],
                help=(
                    "La imputación se aplica únicamente a columnas numéricas. "
                    "Si elige 'No imputar', el modelo descartará las filas incompletas."
                ),
            )
            eliminar_atipicos = st.checkbox(
                "Eliminar valores atípicos detectados con la regla IQR (1.5 × IQR)",
                value=False,
            )
            aplicar = st.form_submit_button("Aplicar limpieza", type="primary")

        if aplicar:
            limpio, reporte = limpiar_datos(df_crudo, estrategia, eliminar_atipicos)
            st.session_state.df_limpio = limpio
            st.session_state.reporte_limpieza = reporte
            reiniciar_modelo()
            st.success(
                f"Limpieza aplicada: {reporte['filas_iniciales']} → "
                f"{reporte['filas_finales']} filas."
            )

        reporte = st.session_state.reporte_limpieza
        if reporte is not None:
            st.markdown("**Resumen de la limpieza aplicada**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filas iniciales", reporte["filas_iniciales"])
            m2.metric("Filas finales", reporte["filas_finales"])
            m3.metric("Faltantes restantes", reporte["faltantes_finales"])
            m4.metric("Atípicos eliminados", reporte["atipicos_eliminados"])
            if reporte["columnas_convertidas"]:
                st.info(
                    "Columnas convertidas a numérico: "
                    + ", ".join(reporte["columnas_convertidas"])
                )
            if st.button("Quitar limpieza y volver a los datos originales"):
                st.session_state.df_limpio = None
                st.session_state.reporte_limpieza = None
                reiniciar_modelo()
                st.rerun()

        st.subheader("Detección de valores atípicos (IQR)")
        numericas_crudas = columnas_numericas(df_crudo)
        if numericas_crudas:
            atipicos = detectar_atipicos_iqr(df_crudo, numericas_crudas)
            st.dataframe(
                atipicos.style.format({
                    "Límite inferior": "{:,.3f}",
                    "Límite superior": "{:,.3f}",
                    "Porcentaje": "{:.2f}%",
                }),
                width="stretch",
            )
            st.caption(
                "Un valor es atípico si está fuera del intervalo "
                "[Q1 − 1.5·IQR, Q3 + 1.5·IQR]."
            )
        else:
            st.warning("No se detectaron columnas numéricas en el archivo.")

        if st.session_state.df_limpio is not None:
            st.subheader("Datos limpios en uso")
            st.dataframe(st.session_state.df_limpio.head(10), width="stretch")


# ----------------------------------------------------------------------
# 2. Estadísticas descriptivas
# ----------------------------------------------------------------------
elif menu.startswith("2"):
    st.header("2. Estadísticas descriptivas")
    df = datos_trabajo()
    if df is None:
        st.warning("Primero cargue un archivo CSV en la sección 1.")
    else:
        numericas = columnas_numericas(df)
        if not numericas:
            st.error("El conjunto de datos no contiene variables numéricas.")
        else:
            if st.session_state.df_limpio is not None:
                st.caption("Se están usando los datos después de la limpieza.")
            st.subheader("Resumen descriptivo")
            descriptivas = estadisticas_descriptivas(df, numericas)
            st.dataframe(
                descriptivas.style.format({
                    "Media": "{:,.4f}",
                    "Varianza": "{:,.4f}",
                    "Desv. estándar": "{:,.4f}",
                    "Mínimo": "{:,.3f}",
                    "Q1": "{:,.3f}",
                    "Mediana": "{:,.3f}",
                    "Q3": "{:,.3f}",
                    "Máximo": "{:,.3f}",
                }),
                width="stretch",
            )
            st.caption(
                "La varianza y la desviación estándar se calculan con corrección "
                "de muestra (n − 1)."
            )

            st.subheader("Matriz de correlaciones")
            variables_relevantes = [
                c for c in [y_sel, *xs_sel] if c is not None and c in df.columns
            ]
            if len(variables_relevantes) >= 2:
                fig_corr = grafico_correlacion(df, variables_relevantes)
                st.pyplot(fig_corr, width="stretch")
                st.download_button(
                    "Descargar matriz de correlaciones (PNG)",
                    figura_a_png(fig_corr),
                    file_name="matriz_correlaciones.png",
                    mime="image/png",
                    key="desc_correlaciones",
                )
                plt.close(fig_corr)
                st.dataframe(
                    df[variables_relevantes].apply(pd.to_numeric, errors="coerce")
                    .corr()
                    .style.format("{:,.3f}"),
                    width="stretch",
                )
            else:
                st.info(
                    "Seleccione al menos dos variables en la barra lateral para "
                    "ver la matriz de correlaciones."
                )


# ----------------------------------------------------------------------
# 3. Estimación del modelo
# ----------------------------------------------------------------------
elif menu.startswith("3"):
    st.header("3. Estimación del modelo de regresión")
    df = datos_trabajo()
    st.write(
        "El modelo estimado es "
        r"$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_k X_k + \varepsilon$ "
        "mediante mínimos cuadrados ordinarios (MCO)."
    )

    if df is None:
        st.warning("Primero cargue un archivo CSV en la sección 1.")
    elif not y_sel or not xs_sel:
        st.warning(
            "Seleccione una variable dependiente y al menos una variable "
            "independiente en la barra lateral."
        )
    else:
        st.info(
            f"**Y:** {y_sel}  \n"
            f"**X:** {', '.join(xs_sel)}  \n"
            f"**Observaciones disponibles:** {len(df)}"
        )

        if st.button("Estimar modelo por MCO", type="primary"):
            try:
                modelo = ajustar_regresion(df, y_sel, xs_sel)
            except Exception as error:
                st.error(f"No se pudo estimar el modelo: {error}")
            else:
                st.session_state.modelo = modelo
                st.session_state.pruebas = None
                st.session_state.reporte_texto = None
                st.success("Modelo estimado correctamente.")

        modelo = st.session_state.modelo
        if modelo is not None:
            mostrar_advertencia_seleccion(modelo, y_sel, xs_sel)

            st.subheader("Ecuación estimada")
            st.code(ecuacion_modelo(modelo), language="text")

            if modelo["advertencia"]:
                st.warning(modelo["advertencia"])

            st.subheader("Bondad de ajuste y significancia global")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R²", numero(modelo["r2"]))
            m2.metric("R² ajustado", numero(modelo["r2_ajustado"]))
            m3.metric("Estadístico F", numero(modelo["f"]))
            m4.metric("p-valor del modelo", numero(modelo["p_f"]))

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Observaciones", modelo["n"])
            m6.metric("Variables independientes", modelo["k"])
            m7.metric("Error estándar de la regresión", numero(modelo["error_estandar_regresion"]))
            m8.metric("Durbin-Watson", numero(modelo["durbin_watson"]))

            if np.isfinite(modelo["p_f"]):
                if modelo["p_f"] < 0.05:
                    st.success(
                        "El modelo global es estadísticamente significativo "
                        "(p < 0.05): las variables independientes en conjunto "
                        "explican a la variable dependiente."
                    )
                else:
                    st.warning(
                        "El modelo global no es estadísticamente significativo "
                        "(p ≥ 0.05): no hay evidencia suficiente de una relación "
                        "lineal conjunta."
                    )

            if modelo["filas_descartadas"] > 0:
                st.caption(
                    f"Se descartaron {modelo['filas_descartadas']} filas con "
                    "valores faltantes o no numéricos en las variables del modelo."
                )

            st.subheader("Coeficientes estimados")
            st.dataframe(tabla_coeficientes(modelo), width="stretch")
            st.caption(
                "Significancia: *** p < 0.01 · ** p < 0.05 · * p < 0.10 · "
                "ns = no significativo. La columna t contrasta H₀: βⱼ = 0."
            )

            with st.expander("Ver errores estándar, t y p-valores en formato de texto"):
                st.text(
                    modelo["coeficientes"].to_string(
                        index=False,
                        float_format=lambda v: f"{v:,.4f}",
                    )
                )

            st.divider()
            if st.button("Borrar modelo actual y sus gráficos"):
                reiniciar_modelo()
                st.rerun()


# ----------------------------------------------------------------------
# 4. Validación del modelo
# ----------------------------------------------------------------------
elif menu.startswith("4"):
    st.header("4. Validación del modelo")
    modelo = st.session_state.modelo
    if modelo is None:
        st.warning(
            "Aún no hay un modelo estimado. Vaya a la sección 3 y presione "
            "'Estimar modelo por MCO'."
        )
    else:
        mostrar_advertencia_seleccion(modelo, y_sel, xs_sel)
        pruebas = ejecutar_pruebas(modelo)
        st.session_state.pruebas = pruebas

        st.caption(
            "Nivel de significancia utilizado: α = 0.05. "
            "Los p-valores menores que α llevan a rechazar la hipótesis nula."
        )

        st.subheader("4.1. Normalidad de los residuos")
        normalidad = pruebas["normalidad"]
        if not normalidad.get("aplicable"):
            st.warning(normalidad.get("mensaje", "Prueba no aplicable."))
        else:
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Jarque-Bera", numero(normalidad["jb"]))
            n2.metric("p-valor JB", numero(normalidad["p_jb"]))
            n3.metric("Kolmogorov-Smirnov", numero(normalidad["ks"]))
            n4.metric("p-valor KS", numero(normalidad["p_ks"]))

            if normalidad["rechaza_jb"]:
                st.warning(normalidad["conclusion_jb"])
            else:
                st.success(normalidad["conclusion_jb"])
            st.caption(
                f"Jarque-Bera contrasta sesgo = 0 y curtosis = 3. "
                f"Sesgo estimado: {normalidad['sesgo']:.4f}; "
                f"curtosis estimada: {normalidad['curtosis']:.4f}. "
                f"Kolmogorov-Smirnov (aproximado, parámetros estimados): "
                f"{normalidad['conclusion_ks']}"
            )

        st.subheader("4.2. Autocorrelación (Durbin-Watson)")
        dw = pruebas["durbin_watson"]
        st.metric("Estadístico Durbin-Watson", numero(dw["estadistico"]))
        if dw["nivel"] == "Sin evidencia de autocorrelación":
            st.success(dw["conclusion"])
        else:
            st.warning(dw["conclusion"])
        st.caption(
            "Interpretación práctica: DW ≈ 2 indica ausencia de autocorrelación; "
            "DW < 1.5 sugiere autocorrelación positiva y DW > 2.5, negativa."
        )

        st.subheader("4.3. Homocedasticidad")
        bp = pruebas["breusch_pagan"]
        white = pruebas["white"]
        cbp, cwhite = st.columns(2)
        with cbp:
            st.markdown("**Breusch-Pagan**")
            if bp.get("aplicable"):
                st.metric("Estadístico LM", numero(bp["estadistico"]))
                st.metric("p-valor", numero(bp["p"]))
                st.metric("Grados de libertad", bp["gl"])
                if bp["rechaza"]:
                    st.warning(bp["conclusion"])
                else:
                    st.success(bp["conclusion"])
            else:
                st.info(bp.get("mensaje", "Prueba no aplicable."))
        with cwhite:
            st.markdown("**White**")
            if white.get("aplicable"):
                st.metric("Estadístico LM", numero(white["estadistico"]))
                st.metric("p-valor", numero(white["p"]))
                st.metric("Grados de libertad", white["gl"])
                if white["rechaza"]:
                    st.warning(white["conclusion"])
                else:
                    st.success(white["conclusion"])
            else:
                st.info(white.get("mensaje", "Prueba no aplicable."))

        st.subheader("4.4. Multicolinealidad (VIF)")
        vif = pruebas["vif"]
        st.dataframe(tabla_vif(vif), width="stretch")
        st.caption(
            "VIFᵢ = 1 / (1 − Rᵢ²). Valores mayores que 10 se resaltan en rojo "
            "e indican multicolinealidad problemática."
        )
        if (vif["VIF"] > 10).any() or np.isinf(vif["VIF"]).any():
            st.warning(
                "Existen variables con VIF > 10. Considere eliminar o combinar "
                "las variables independientes redundantes."
            )
        else:
            st.success(
                "Ninguna variable independiente supera el umbral de VIF = 10."
            )


# ----------------------------------------------------------------------
# 5. Gráficos
# ----------------------------------------------------------------------
elif menu.startswith("5"):
    st.header("5. Gráficos del modelo")
    modelo = st.session_state.modelo
    if modelo is None:
        st.warning(
            "Aún no hay un modelo estimado. Vaya a la sección 3 y presione "
            "'Estimar modelo por MCO'."
        )
    else:
        mostrar_advertencia_seleccion(modelo, y_sel, xs_sel)
        df = modelo["datos_modelo"]

        pestanas = st.tabs(["Dispersión Xi vs. Y", "Diagnóstico de residuos"])

        with pestanas[0]:
            st.write(
                "Diagrama de dispersión de cada variable independiente frente "
                "a la variable dependiente, con recta de tendencia."
            )
            columnas = st.columns(2)
            for indice, variable in enumerate(modelo["independientes"]):
                with columnas[indice % 2]:
                    figura = grafico_dispersion(df, modelo["dependiente"], variable)
                    st.pyplot(figura, width="stretch")
                    st.download_button(
                        f"Descargar dispersión {variable} (PNG)",
                        figura_a_png(figura),
                        file_name=f"dispersion_{variable}.png",
                        mime="image/png",
                        key=f"desc_disp_{indice}",
                    )
                    plt.close(figura)

        with pestanas[1]:
            st.write(
                "Gráficos para evaluar normalidad, linealidad y "
                "homocedasticidad de los residuos del modelo actual."
            )
            col1, col2 = st.columns(2)

            with col1:
                figura_hist = histograma_residuos(modelo["residuos"])
                st.pyplot(figura_hist, width="stretch")
                st.download_button(
                    "Descargar histograma (PNG)",
                    figura_a_png(figura_hist),
                    file_name="histograma_residuos.png",
                    mime="image/png",
                    key="desc_hist",
                )
                plt.close(figura_hist)

            with col2:
                figura_qq = grafico_qq(modelo["residuos"])
                st.pyplot(figura_qq, width="stretch")
                st.download_button(
                    "Descargar Q-Q (PNG)",
                    figura_a_png(figura_qq),
                    file_name="qq_residuos.png",
                    mime="image/png",
                    key="desc_qq",
                )
                plt.close(figura_qq)

            figura_res = residuos_estandar_vs_ajustados(
                modelo["ajustados"],
                modelo["residuos_estandarizados"],
            )
            st.pyplot(figura_res, width="stretch")
            st.download_button(
                "Descargar residuos estandarizados (PNG)",
                figura_a_png(figura_res),
                file_name="residuos_estandarizados.png",
                mime="image/png",
                key="desc_residuos",
            )
            plt.close(figura_res)
            st.caption(
                "Las líneas discontinuas rojas en ±2 permiten identificar "
                "observaciones con residuos estandarizados extremos."
            )


# ----------------------------------------------------------------------
# 6. Reporte final
# ----------------------------------------------------------------------
elif menu.startswith("6"):
    st.header("6. Reporte final")
    modelo = st.session_state.modelo
    if modelo is None:
        st.warning(
            "Aún no hay un modelo estimado. Vaya a la sección 3 y presione "
            "'Estimar modelo por MCO'."
        )
    else:
        mostrar_advertencia_seleccion(modelo, y_sel, xs_sel)
        pruebas = st.session_state.pruebas or ejecutar_pruebas(modelo)
        st.session_state.pruebas = pruebas

        if st.button("Generar reporte", type="primary"):
            st.session_state.reporte_texto = generar_reporte_texto(
                st.session_state.info_carga or {},
                datos_trabajo(),
                modelo["dependiente"],
                modelo["independientes"],
                modelo,
                pruebas,
            )

        reporte = st.session_state.reporte_texto
        if reporte:
            st.text_area("Vista previa del reporte", reporte, height=420)
            st.download_button(
                "Descargar reporte (.txt)",
                reporte,
                file_name="reporte_regresion.txt",
                mime="text/plain",
            )
        else:
            st.info("Presione 'Generar reporte' para crear el documento de resultados.")
