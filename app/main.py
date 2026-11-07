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
    proceso_estimacion,
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


def mostrar_proceso_estimacion(modelo: dict) -> None:
    """Explica y muestra los cálculos matriciales que producen la ecuación MCO."""
    proceso = proceso_estimacion(modelo)
    X = proceso["X"]
    y = proceso["y"]
    nombres = ["Intercepto", *modelo["independientes"]]
    n, p = X.shape
    vista = min(n, 8)

    st.markdown("### Proceso de estimación por mínimos cuadrados ordinarios")
    st.caption(
        "El procedimiento siguiente muestra cómo se pasa del modelo propuesto "
        "y de los datos observados a los coeficientes de la ecuación estimada."
    )

    st.markdown("**Paso 1 — Modelo propuesto**")
    st.latex(r"Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_k X_{ik} + \varepsilon_i")
    st.write(
        f"Para este ajuste: n = {n} observaciones, k = {modelo['k']} variables "
        f"independientes y p = k + 1 = {p} parámetros (incluyendo el intercepto)."
    )

    st.markdown("**Paso 2 — Función objetivo de MCO**")
    st.write(
        "MCO elige los coeficientes que minimizan la suma de residuos al cuadrado:"
    )
    st.latex(
        r"SCR(\beta)=\sum_{i=1}^{n}"
        r"\left(Y_i-\beta_0-\beta_1X_{i1}-\cdots-\beta_kX_{ik}\right)^2"
    )
    st.write(
        "Al derivar respecto a cada coeficiente e igualar a cero se obtienen "
        "las ecuaciones normales:"
    )
    st.latex(r"\mathbf{X}^{T}\mathbf{X}\,\hat{\beta}=\mathbf{X}^{T}\mathbf{Y}")

    st.markdown("**Paso 3 — Matriz de diseño X y vector Y**")
    st.write(
        "Cada fila de X representa una observación. La primera columna es el "
        "intercepto (todo 1) y las siguientes son las variables independientes."
    )
    indice_vista = [f"Obs {i}" for i in range(1, vista + 1)]
    col_x, col_y = st.columns([4, 1])
    with col_x:
        st.caption(f"Matriz X: dimensiones {n} × {p}. Primeras {vista} filas.")
        st.dataframe(
            pd.DataFrame(X[:vista], index=indice_vista, columns=nombres)
            .style.format("{:,.4f}"),
            width="stretch",
        )
    with col_y:
        st.caption(f"Vector Y: dimensiones {n} × 1.")
        st.dataframe(
            pd.DataFrame(y[:vista], index=indice_vista, columns=[modelo["dependiente"]])
            .style.format("{:,.4f}"),
            width="stretch",
        )

    st.markdown("**Paso 4 — Productos XᵀX y XᵀY**")
    col_xtx, col_xty = st.columns([3, 1])
    with col_xtx:
        st.caption("Matriz XᵀX")
        st.dataframe(
            pd.DataFrame(proceso["XtX"], index=nombres, columns=nombres)
            .style.format("{:,.4f}"),
            width="stretch",
        )
    with col_xty:
        st.caption("Vector XᵀY")
        st.dataframe(
            pd.DataFrame(proceso["Xty"], index=nombres, columns=["XᵀY"])
            .style.format("{:,.4f}"),
            width="stretch",
        )

    st.markdown("**Paso 5 — Inversa de XᵀX y solución de las ecuaciones normales**")
    st.write(
        f"La matriz XᵀX tiene rango {proceso['rango']}; se utiliza su "
        f"{proceso['metodo_inversa']} para despejar los coeficientes:"
    )
    st.latex(r"\hat{\beta}=(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{Y}")
    st.caption("Matriz (XᵀX)⁻¹")
    st.dataframe(
        pd.DataFrame(proceso["XtX_inversa"], index=nombres, columns=nombres)
        .style.format("{:,.6f}"),
        width="stretch",
    )

    tabla_beta = pd.DataFrame({
        "Variable": nombres,
        "Coeficiente estimado": proceso["beta"],
    })
    st.caption("Resultado del producto (XᵀX)⁻¹XᵀY")
    st.dataframe(
        tabla_beta.style.format({"Coeficiente estimado": "{:,.6f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Paso 6 — Verificación de las ecuaciones normales**")
    st.write(
        "Comprobación numérica: el producto XᵀX·β̂ debe coincidir con XᵀY."
    )
    verificacion = pd.DataFrame({
        "Variable": nombres,
        "XᵀX·β̂": proceso["XtX"] @ proceso["beta"],
        "XᵀY": proceso["Xty"],
        "Diferencia": proceso["verificacion"],
    })
    st.dataframe(
        verificacion.style.format({
            "XᵀX·β̂": "{:,.6f}",
            "XᵀY": "{:,.6f}",
            "Diferencia": "{:.2e}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Error máximo absoluto de la verificación: "
        f"{proceso['error_maximo_verificacion']:.2e}"
    )

    st.markdown("**Paso 7 — Valores ajustados y residuos**")
    st.latex(r"\hat{Y}=\mathbf{X}\hat{\beta}, \qquad e=Y-\hat{Y}")
    tabla_residuos = pd.DataFrame({
        "Observación": np.arange(1, vista + 1),
        "Y observado": y[:vista],
        "Ŷ ajustado": proceso["ajustados"][:vista],
        "Residuo e": proceso["residuos"][:vista],
    })
    st.dataframe(
        tabla_residuos.style.format({
            "Y observado": "{:,.4f}",
            "Ŷ ajustado": "{:,.4f}",
            "Residuo e": "{:,.4f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Suma de cuadrados de residuos (SCR) = {modelo['sse']:,.4f}; "
        f"suma de cuadrados total (SCT) = {modelo['sst']:,.4f}; "
        f"R² = 1 − SCR/SCT = {modelo['r2']:.4f}."
    )

    st.markdown("**Paso 8 — Varianza estimada de los errores**")
    st.write(
        "Se calcula como el producto interno del vector de residuos consigo "
        "mismo, dividido por los grados de libertad residuales:"
    )
    st.latex(
        r"\widehat{\sigma}^{2}="
        r"\frac{\hat{\mathbf{u}}^{T}\hat{\mathbf{u}}}{n-p}="
        r"\frac{\sum_{i=1}^{n}\hat{u}_{i}^{2}}{n-p}"
    )
    st.caption(
        f"Aquí n = {modelo['n']}, k = {modelo['k']} variables independientes, "
        f"y p = k + 1 = {modelo['p']} parámetros estimados (incluido el intercepto). "
        f"Por tanto, n − p = n − k − 1 = {modelo['gl_residual']} grados de libertad. "
        "Si en la notación de su curso k representa todos los parámetros, incluido "
        "el intercepto, esta misma expresión se escribe n − k."
    )
    sse_residuos = float(proceso["residuos"] @ proceso["residuos"])
    varianza_estimada = sse_residuos / modelo["gl_residual"]
    st.code(
        f"u_hatᵀ u_hat = {sse_residuos:,.6f}\n"
        f"n − p = {modelo['n']} − {modelo['p']} = {modelo['gl_residual']}\n"
        f"sigma_hat² = {sse_residuos:,.6f} / {modelo['gl_residual']} "
        f"= {varianza_estimada:,.6f}",
        language="text",
    )

    st.markdown("**Paso 9 — Matriz de varianzas y covarianzas de los coeficientes**")
    st.write(
        "La matriz de varianzas-covarianzas de β̂ se obtiene multiplicando la "
        "varianza estimada del error por la inversa de XᵀX:"
    )
    st.latex(
        r"\widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}})="
        r"\widehat{\sigma}^{2}(\mathbf{X}^{T}\mathbf{X})^{-1}"
    )
    matriz_covarianza = varianza_estimada * proceso["XtX_inversa"]
    st.dataframe(
        pd.DataFrame(matriz_covarianza, index=nombres, columns=nombres)
        .style.format("{:,.6f}"),
        width="stretch",
    )
    st.caption(
        "Los elementos de la diagonal son las varianzas estimadas de cada "
        "coeficiente; los elementos fuera de la diagonal son covarianzas."
    )

    st.markdown("**Paso 10 — Desviaciones estándar y estadísticos t de Student**")
    st.write(
        "La desviación estándar de cada coeficiente es la raíz cuadrada del "
        "elemento correspondiente de la diagonal de la matriz anterior. "
        "Luego, el estadístico t contrasta H₀: βⱼ = 0."
    )
    st.latex(
        r"SE(\hat{\beta}_{j})="
        r"\sqrt{\widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}})_{jj}},"
        r"\qquad t_{j}=\frac{\hat{\beta}_{j}}{SE(\hat{\beta}_{j})}"
    )
    errores_estandar = np.sqrt(np.maximum(np.diag(matriz_covarianza), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_calculados = proceso["beta"] / errores_estandar
    tabla_inferencia = pd.DataFrame({
        "Variable / coeficiente": nombres,
        "β̂ (coeficiente)": proceso["beta"],
        "Varianza (diagonal)": np.diag(matriz_covarianza),
        "Desviación estándar": errores_estandar,
        "Cálculo de t = β̂ / SE": [
            f"{beta:,.6f} / {se:,.6f}"
            for beta, se in zip(proceso["beta"], errores_estandar)
        ],
        "t de Student": t_calculados,
    })
    st.caption(
        f"Variable dependiente: {modelo['dependiente']}. Las filas corresponden "
        "al intercepto (β̂₀) y a cada variable independiente."
    )
    st.dataframe(
        tabla_inferencia.style.format({
            "β̂ (coeficiente)": "{:,.6f}",
            "Varianza (diagonal)": "{:,.6f}",
            "Desviación estándar": "{:,.6f}",
            "t de Student": "{:,.6f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Los t se comparan con una distribución t de Student con "
        f"{modelo['gl_residual']} grados de libertad residuales. "
        "La tabla de coeficientes que sigue incluye también sus p-valores."
    )

    st.success("Ecuación estimada final")
    st.code(ecuacion_modelo(modelo), language="text")



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

            with st.expander("Proceso de estimación: cómo se obtuvo la ecuación", expanded=True):
                mostrar_proceso_estimacion(modelo)

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
