"""
regresion.py
Núcleo estadístico del Sistema de Regresión Lineal Multivariada.

Implementa, con NumPy/Pandas y funciones matemáticas propias (sin
statsmodels ni scipy), los siguientes elementos de la guía:

- Estadísticas descriptivas y detección de valores atípicos.
- Estimación de mínimos cuadrados ordinarios (MCO).
- Coeficientes, errores estándar, estadístico t y p-valores.
- Pruebas de normalidad (Jarque-Bera y Kolmogorov-Smirnov).
- Durbin-Watson para autocorrelación.
- Breusch-Pagan y White para homocedasticidad.
- VIF para multicolinealidad.

Práctica de Investigación Operativa II
"""

from __future__ import annotations

import codecs
import csv
import io
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

ALFA = 0.05


# ======================================================================
# Utilidades matemáticas
# ======================================================================
def _betacf(a: float, b: float, x: float) -> float:
    """Fracción continua para la función beta incompleta (Lentz)."""
    max_iter = 300
    eps = 3e-14
    fpmin = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h


def beta_incompleta_regularizada(a: float, b: float, x: float) -> float:
    """I_x(a, b), función beta incompleta regularizada."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def p_valor_t(t: float, gl: int) -> float:
    """p-valor bilateral de una distribución t de Student."""
    if not np.isfinite(t) or gl <= 0:
        return float("nan")
    x = gl / (gl + t * t)
    return float(beta_incompleta_regularizada(gl / 2.0, 0.5, x))


def p_valor_f(f: float, gl1: int, gl2: int) -> float:
    """p-valor de la cola derecha de una distribución F."""
    if not np.isfinite(f) or f <= 0 or gl1 <= 0 or gl2 <= 0:
        return 1.0 if f == 0 else float("nan")
    x = gl2 / (gl2 + gl1 * f)
    return float(beta_incompleta_regularizada(gl2 / 2.0, gl1 / 2.0, x))


def _gamma_inferior_regularizada(a: float, x: float) -> float:
    """P(a, x): función gamma incompleta regularizada inferior."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        # Desarrollo en serie.
        ap = a
        suma = 1.0 / a
        delta = suma
        for _ in range(300):
            ap += 1.0
            delta *= x / ap
            suma += delta
            if abs(delta) < abs(suma) * 1e-14:
                break
        return suma * math.exp(-x + a * math.log(x) - math.lgamma(a))

    # Fracción continua para la gamma superior.
    fpmin = 1e-300
    eps = 3e-14
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 301):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    superior = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - superior


def p_valor_chi2(estadistico: float, gl: int) -> float:
    """p-valor de la cola derecha de una distribución chi-cuadrado."""
    if not np.isfinite(estadistico) or gl <= 0:
        return float("nan")
    if estadistico <= 0:
        return 1.0
    return float(1.0 - _gamma_inferior_regularizada(gl / 2.0, estadistico / 2.0))


def _r2_auxiliar(X: np.ndarray, objetivo: np.ndarray) -> float:
    """R² de una regresión auxiliar usada en pruebas diagnósticas."""
    beta, *_ = np.linalg.lstsq(X, objetivo, rcond=None)
    residuos = objetivo - X @ beta
    sse = float(residuos @ residuos)
    media = float(np.mean(objetivo))
    sst = float(np.sum((objetivo - media) ** 2))
    if sst <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - sse / sst))


# ======================================================================
# Carga y limpieza de datos
# ======================================================================
def detectar_delimitador(texto: str) -> str:
    """Detecta automáticamente el delimitador de un CSV."""
    muestra = texto[:8192]
    try:
        dialecto = csv.Sniffer().sniff(muestra, delimiters=",;\t|")
        return dialecto.delimiter
    except csv.Error:
        # Alternativa: elegir el delimitador más frecuente en la cabecera.
        cabecera = muestra.splitlines()[0] if muestra.splitlines() else ""
        candidatos = {",": cabecera.count(","), ";": cabecera.count(";"),
                      "\t": cabecera.count("\t"), "|": cabecera.count("|")}
        delimitador = max(candidatos, key=candidatos.get)
        return delimitador if candidatos[delimitador] > 0 else ","


def leer_csv_desde_bytes(contenido: bytes) -> tuple[pd.DataFrame, str, str]:
    """Lee un CSV subido por el usuario y devuelve (dataframe, delimitador, codificación)."""
    texto = None
    codificacion = "utf-8"
    for cod in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            texto = contenido.decode(cod)
            if cod == "utf-8-sig":
                codificacion = "utf-8-sig" if contenido.startswith(codecs.BOM_UTF8) else "utf-8"
            else:
                codificacion = cod
            break
        except UnicodeDecodeError:
            continue
    if texto is None:
        raise ValueError("No fue posible decodificar el archivo. Use codificación UTF-8.")

    delimitador = detectar_delimitador(texto)
    df = pd.read_csv(io.StringIO(texto), sep=delimitador)
    # Elimina columnas completamente vacías generadas por separadores finales.
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    nombres = list(df.columns)
    if not nombres or any(not nombre for nombre in nombres):
        raise ValueError("El archivo contiene nombres de columna vacíos.")
    if len(set(nombres)) != len(nombres):
        raise ValueError("El archivo contiene nombres de columna duplicados.")
    if df.empty:
        raise ValueError("El archivo no contiene filas de datos.")
    return df, delimitador, codificacion


def convertir_columnas_numericas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convierte a numérico las columnas que tengan al menos un valor parseable."""
    trabajo = df.copy()
    convertidas = []
    for columna in trabajo.columns:
        if pd.api.types.is_numeric_dtype(trabajo[columna]):
            continue
        convertida = pd.to_numeric(trabajo[columna], errors="coerce")
        if convertida.notna().sum() > 0:
            trabajo[columna] = convertida
            convertidas.append(columna)
    return trabajo, convertidas


def columnas_numericas(df: pd.DataFrame) -> list[str]:
    """Nombres de las columnas numéricas."""
    return list(df.select_dtypes(include=[np.number]).columns)


def estadisticas_descriptivas(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """Media, varianza, desviación, mínimo, cuartiles y máximo por variable."""
    columnas = columnas or columnas_numericas(df)
    filas = []
    for columna in columnas:
        serie = pd.to_numeric(df[columna], errors="coerce")
        validos = serie.dropna()
        if validos.empty:
            filas.append({
                "Variable": columna, "n": 0, "Faltantes": int(serie.isna().sum()),
                "Media": np.nan, "Varianza": np.nan, "Desv. estándar": np.nan,
                "Mínimo": np.nan, "Q1": np.nan, "Mediana": np.nan,
                "Q3": np.nan, "Máximo": np.nan,
            })
            continue
        filas.append({
            "Variable": columna,
            "n": int(validos.count()),
            "Faltantes": int(serie.isna().sum()),
            "Media": float(validos.mean()),
            "Varianza": float(validos.var(ddof=1)) if len(validos) > 1 else 0.0,
            "Desv. estándar": float(validos.std(ddof=1)) if len(validos) > 1 else 0.0,
            "Mínimo": float(validos.min()),
            "Q1": float(validos.quantile(0.25)),
            "Mediana": float(validos.median()),
            "Q3": float(validos.quantile(0.75)),
            "Máximo": float(validos.max()),
        })
    return pd.DataFrame(filas).set_index("Variable") if filas else pd.DataFrame()


def detectar_atipicos_iqr(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """Cuenta valores atípicos por la regla de 1.5 veces el rango intercuartílico."""
    columnas = columnas or columnas_numericas(df)
    filas = []
    for columna in columnas:
        serie = pd.to_numeric(df[columna], errors="coerce").dropna()
        if serie.empty:
            continue
        q1 = float(serie.quantile(0.25))
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        mascara = (serie < limite_inferior) | (serie > limite_superior)
        filas.append({
            "Variable": columna,
            "Límite inferior": limite_inferior,
            "Límite superior": limite_superior,
            "Atípicos": int(mascara.sum()),
            "Porcentaje": 100.0 * float(mascara.mean()),
        })
    return pd.DataFrame(filas).set_index("Variable") if filas else pd.DataFrame()


def limpiar_datos(
    df: pd.DataFrame,
    estrategia_faltantes: str = "Eliminar filas con faltantes",
    eliminar_atipicos: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Limpia el dataframe según las opciones elegidas por el usuario.

    Estrategias de faltantes:
    - "Eliminar filas con faltantes"
    - "Imputar con la media"
    - "Imputar con la mediana"
    - "No imputar"
    """
    trabajo, convertidas = convertir_columnas_numericas(df)
    numericas = columnas_numericas(trabajo)
    filas_iniciales = len(trabajo)
    faltantes_iniciales = int(trabajo[numericas].isna().sum().sum()) if numericas else 0
    atipicos_iniciales = (
        int(detectar_atipicos_iqr(trabajo, numericas)["Atípicos"].sum())
        if numericas else 0
    )

    if numericas:
        if estrategia_faltantes == "Eliminar filas con faltantes":
            trabajo = trabajo.dropna(subset=numericas).reset_index(drop=True)
        elif estrategia_faltantes == "Imputar con la media":
            trabajo[numericas] = trabajo[numericas].fillna(trabajo[numericas].mean())
        elif estrategia_faltantes == "Imputar con la mediana":
            trabajo[numericas] = trabajo[numericas].fillna(trabajo[numericas].median())
        # "No imputar" deja los valores faltantes para que el modelo los descarte.

    atipicos_eliminados = 0
    if eliminar_atipicos and numericas:
        mascara = pd.Series(True, index=trabajo.index)
        for columna in numericas:
            serie = trabajo[columna]
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            mascara &= serie.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        atipicos_eliminados = int((~mascara).sum())
        trabajo = trabajo.loc[mascara].reset_index(drop=True)

    trabajo = trabajo.reset_index(drop=True)
    reporte = {
        "filas_iniciales": filas_iniciales,
        "filas_finales": len(trabajo),
        "filas_eliminadas": filas_iniciales - len(trabajo),
        "faltantes_iniciales": faltantes_iniciales,
        "faltantes_finales": int(trabajo[numericas].isna().sum().sum()) if numericas else 0,
        "atipicos_detectados": atipicos_iniciales,
        "atipicos_eliminados": atipicos_eliminados,
        "columnas_convertidas": convertidas,
        "columnas_numericas": numericas,
    }
    return trabajo, reporte


# ======================================================================
# Estimación del modelo de regresión
# ======================================================================
def _significancia(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "ns"


def ajustar_regresion(
    df: pd.DataFrame,
    dependiente: str,
    independientes: list[str],
) -> dict:
    """
    Estima Y = beta0 + beta1 X1 + ... + betak Xk + epsilon por MCO.

    Devuelve coeficientes, inferencia, residuos, bondad de ajuste y
    elementos necesarios para las pruebas de validación y los gráficos.
    """
    if not independientes:
        raise ValueError("Seleccione al menos una variable independiente.")
    if dependiente in independientes:
        raise ValueError("La variable dependiente no puede estar entre las independientes.")
    faltantes = [c for c in [dependiente, *independientes] if c not in df.columns]
    if faltantes:
        raise ValueError(f"No existen las columnas: {', '.join(faltantes)}")

    datos = df[[dependiente, *independientes]].apply(pd.to_numeric, errors="coerce")
    mascara = np.isfinite(datos.to_numpy(dtype=float)).all(axis=1)
    datos = datos.loc[mascara]
    filas_descartadas = int((~mascara).sum())

    y = datos[dependiente].to_numpy(dtype=float)
    X_sin_intercepto = datos[independientes].to_numpy(dtype=float)
    n = len(y)
    k = len(independientes)
    p = k + 1

    if n <= p:
        raise ValueError(
            f"Observaciones insuficientes: n={n} y parámetros={p}. "
            "Se necesitan más filas que variables independientes + intercepto."
        )

    X = np.column_stack([np.ones(n), X_sin_intercepto])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ajustados = X @ beta
    residuos = y - ajustados

    sse = float(residuos @ residuos)
    media_y = float(np.mean(y))
    sst = float(np.sum((y - media_y) ** 2))
    gl_residual = n - p
    sigma2 = sse / gl_residual if gl_residual > 0 else float("nan")
    error_estandar_regresion = math.sqrt(sigma2) if sigma2 >= 0 else float("nan")

    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    r2_ajustado = (
        1.0 - (1.0 - r2) * (n - 1) / gl_residual
        if np.isfinite(r2) and gl_residual > 0 else float("nan")
    )

    # Matriz de covarianzas de los estimadores.
    xtx_inversa = np.linalg.pinv(X.T @ X)
    cov_beta = sigma2 * xtx_inversa if np.isfinite(sigma2) else np.full((p, p), np.nan)
    errores_estandar = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        estadisticos_t = beta / errores_estandar
    p_valores = np.array([p_valor_t(float(t), gl_residual) for t in estadisticos_t])

    if k > 0 and gl_residual > 0 and sse > 0:
        f_estadistico = ((sst - sse) / k) / (sse / gl_residual)
        p_f = p_valor_f(float(f_estadistico), k, gl_residual)
    else:
        f_estadistico = float("nan")
        p_f = float("nan")

    dw = durbin_watson(residuos)
    rmse = math.sqrt(sse / n)
    mae = float(np.mean(np.abs(residuos)))

    # Residuos estandarizados (studentizados internamente).
    leverage = np.clip(np.sum((X @ xtx_inversa) * X, axis=1), 0.0, 1.0 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        residuos_estandarizados = residuos / np.sqrt(sigma2 * (1.0 - leverage))

    nombres = ["Intercepto", *independientes]
    coeficientes = pd.DataFrame({
        "Variable": nombres,
        "Coeficiente": beta,
        "Error estándar": errores_estandar,
        "Estadístico t": estadisticos_t,
        "p-valor": p_valores,
    })
    coeficientes["Significancia"] = coeficientes["p-valor"].apply(_significancia)

    rango = int(np.linalg.matrix_rank(X))
    advertencia = None
    if rango < p:
        advertencia = (
            "La matriz de diseño es singular (posible variable constante o "
            "multicolinealidad perfecta). Los resultados deben interpretarse con cautela."
        )

    return {
        "dependiente": dependiente,
        "independientes": list(independientes),
        "n": n,
        "k": k,
        "p": p,
        "filas_descartadas": filas_descartadas,
        "beta": beta,
        "errores_estandar": errores_estandar,
        "estadisticos_t": estadisticos_t,
        "p_valores": p_valores,
        "coeficientes": coeficientes,
        "X": X,
        "y": y,
        "ajustados": ajustados,
        "residuos": residuos,
        "residuos_estandarizados": residuos_estandarizados,
        "leverage": leverage,
        "sse": sse,
        "sst": sst,
        "gl_residual": gl_residual,
        "sigma2": sigma2,
        "error_estandar_regresion": error_estandar_regresion,
        "r2": r2,
        "r2_ajustado": r2_ajustado,
        "f": f_estadistico,
        "p_f": p_f,
        "durbin_watson": dw,
        "rmse": rmse,
        "mae": mae,
        "rango": rango,
        "advertencia": advertencia,
        "datos_modelo": datos,
    }


def ecuacion_modelo(modelo: dict) -> str:
    """Representación legible de la ecuación estimada."""
    terminos = [f"{modelo['beta'][0]:.4f}"]
    for nombre, coeficiente in zip(modelo["independientes"], modelo["beta"][1:]):
        signo = "+" if coeficiente >= 0 else "-"
        terminos.append(f"{signo} {abs(coeficiente):.4f}·{nombre}")
    return f"{modelo['dependiente']} = " + " ".join(terminos)


def proceso_estimacion(modelo: dict) -> dict:
    """
    Calcula y expone los productos matriciales intermedios del MCO:

        beta = (X'X)^(-1) X'Y

    Se devuelven las matrices X'X, X'Y, la inversa (o pseudoinversa) y
    los valores ajustados/residuos para mostrar el procedimiento en la
    interfaz.
    """
    X = np.asarray(modelo["X"], dtype=float)
    y = np.asarray(modelo["y"], dtype=float)
    XtX = X.T @ X
    Xty = X.T @ y
    rango = int(np.linalg.matrix_rank(XtX))
    if rango == XtX.shape[0]:
        XtX_inversa = np.linalg.inv(XtX)
        metodo_inversa = "inversa"
    else:
        XtX_inversa = np.linalg.pinv(XtX)
        metodo_inversa = "pseudoinversa"

    beta = XtX_inversa @ Xty
    ajustados = X @ beta
    residuos = y - ajustados
    verificacion = XtX @ beta - Xty

    return {
        "X": X,
        "y": y,
        "XtX": XtX,
        "Xty": Xty,
        "XtX_inversa": XtX_inversa,
        "beta": beta,
        "ajustados": ajustados,
        "residuos": residuos,
        "verificacion": verificacion,
        "error_maximo_verificacion": float(np.max(np.abs(verificacion))),
        "metodo_inversa": metodo_inversa,
        "rango": rango,
    }


# ======================================================================
# Pruebas de validación
# ======================================================================
def durbin_watson(residuos: np.ndarray) -> float:
    """Estadístico Durbin-Watson para autocorrelación de primer orden."""
    residuos = np.asarray(residuos, dtype=float)
    if len(residuos) < 2:
        return float("nan")
    denominador = float(residuos @ residuos)
    if denominador <= 0:
        return float("nan")
    diferencias = np.diff(residuos)
    return float(diferencias @ diferencias / denominador)


def interpretar_durbin_watson(dw: float) -> tuple[str, str]:
    """Devuelve (nivel, conclusión) para un valor de Durbin-Watson."""
    if not np.isfinite(dw):
        return "No concluyente", "No fue posible calcular el estadístico."
    if dw < 1.5:
        return "Autocorrelación positiva", (
            "El estadístico es menor que 1.5: hay evidencia de autocorrelación "
            "positiva de primer orden."
        )
    if dw > 2.5:
        return "Autocorrelación negativa", (
            "El estadístico es mayor que 2.5: hay evidencia de autocorrelación "
            "negativa de primer orden."
        )
    return "Sin evidencia de autocorrelación", (
        "El estadístico se encuentra en la zona central (entre 1.5 y 2.5): "
        "no hay evidencia clara de autocorrelación de primer orden."
    )


def prueba_normalidad(residuos: np.ndarray) -> dict:
    """
    Pruebas de normalidad sobre los residuos.

    Incluye Jarque-Bera (principal) y Kolmogorov-Smirnov con parámetros
    estimados (aproximada). El p-valor de Jarque-Bera es exacto para gl=2.
    """
    residuos = np.asarray(residuos, dtype=float)
    n = len(residuos)
    if n < 3:
        return {
            "aplicable": False,
            "mensaje": "Se requieren al menos 3 residuos para la prueba de normalidad.",
        }

    media = float(np.mean(residuos))
    desv = float(np.std(residuos, ddof=1))
    if desv <= 0:
        return {
            "aplicable": False,
            "mensaje": "Los residuos tienen desviación estándar cero; la prueba no es aplicable.",
        }

    z = (residuos - media) / desv
    sesgo = float(np.mean(z ** 3))
    curtosis = float(np.mean(z ** 4))  # curtosis (no exceso)
    jb = n / 6.0 * (sesgo ** 2 + (curtosis - 3.0) ** 2 / 4.0)
    p_jb = float(math.exp(-jb / 2.0))  # chi-cuadrado con 2 gl

    # Kolmogorov-Smirnov con media y desviación estimadas (Lilliefors aproximado).
    ordenados = np.sort(z)
    normal = NormalDist()
    cdf = np.array([normal.cdf(float(valor)) for valor in ordenados])
    indices = np.arange(1, n + 1, dtype=float)
    d_mas = float(np.max(indices / n - cdf))
    d_menos = float(np.max(cdf - (indices - 1) / n))
    d_ks = max(d_mas, d_menos)
    # Distribución asintótica de Kolmogorov.
    suma = 0.0
    for j in range(1, 101):
        suma += ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * n * d_ks * d_ks)
    p_ks = min(1.0, max(0.0, 2.0 * suma))

    rechaza_jb = p_jb < ALFA
    rechaza_ks = p_ks < ALFA
    return {
        "aplicable": True,
        "n": n,
        "sesgo": sesgo,
        "curtosis": curtosis,
        "jb": jb,
        "p_jb": p_jb,
        "ks": d_ks,
        "p_ks": p_ks,
        "rechaza_jb": rechaza_jb,
        "rechaza_ks": rechaza_ks,
        "conclusion_jb": (
            "Se rechaza la hipótesis de normalidad de los residuos (p < 0.05)."
            if rechaza_jb else
            "No se rechaza la hipótesis de normalidad de los residuos (p ≥ 0.05)."
        ),
        "conclusion_ks": (
            "Se rechaza la hipótesis de normalidad de los residuos (p < 0.05)."
            if rechaza_ks else
            "No se rechaza la hipótesis de normalidad de los residuos (p ≥ 0.05)."
        ),
    }


def prueba_breusch_pagan(modelo: dict) -> dict:
    """Prueba de Breusch-Pagan para homocedasticidad."""
    X = modelo["X"]
    residuos = modelo["residuos"]
    n = modelo["n"]
    k = modelo["k"]
    if n <= k + 2:
        return {"aplicable": False, "mensaje": "Observaciones insuficientes para la prueba."}

    objetivo = residuos ** 2
    r2_aux = _r2_auxiliar(X, objetivo)
    lm = n * r2_aux
    p = p_valor_chi2(lm, k)
    return {
        "aplicable": True,
        "estadistico": lm,
        "gl": k,
        "p": p,
        "r2_aux": r2_aux,
        "rechaza": p < ALFA,
        "conclusion": (
            "Se rechaza la homocedasticidad: hay evidencia de heterocedasticidad (p < 0.05)."
            if p < ALFA else
            "No se rechaza la homocedasticidad: no hay evidencia suficiente de "
            "heterocedasticidad (p ≥ 0.05)."
        ),
    }


def prueba_white(modelo: dict) -> dict:
    """Prueba de White (términos cuadráticos y productos cruzados)."""
    X_sin_intercepto = modelo["X"][:, 1:]
    residuos = modelo["residuos"]
    n, k = X_sin_intercepto.shape
    if k == 0:
        return {"aplicable": False, "mensaje": "No hay variables independientes."}

    columnas = [np.ones(n), *[X_sin_intercepto[:, j] for j in range(k)]]
    columnas.extend([X_sin_intercepto[:, j] ** 2 for j in range(k)])
    for j in range(k):
        for l in range(j + 1, k):
            columnas.append(X_sin_intercepto[:, j] * X_sin_intercepto[:, l])
    X_aux = np.column_stack(columnas)
    gl = int(np.linalg.matrix_rank(X_aux)) - 1
    if gl <= 0 or n <= gl + 1:
        return {
            "aplicable": False,
            "mensaje": (
                "No hay observaciones suficientes para la prueba de White "
                f"(n={n}, términos auxiliares={gl})."
            ),
        }

    objetivo = residuos ** 2
    r2_aux = _r2_auxiliar(X_aux, objetivo)
    lm = n * r2_aux
    p = p_valor_chi2(lm, gl)
    return {
        "aplicable": True,
        "estadistico": lm,
        "gl": gl,
        "p": p,
        "r2_aux": r2_aux,
        "rechaza": p < ALFA,
        "conclusion": (
            "Se rechaza la homocedasticidad: hay evidencia de heterocedasticidad (p < 0.05)."
            if p < ALFA else
            "No se rechaza la homocedasticidad: no hay evidencia suficiente de "
            "heterocedasticidad (p ≥ 0.05)."
        ),
    }


def calcular_vif(modelo: dict) -> pd.DataFrame:
    """VIF de cada variable independiente mediante regresiones auxiliares."""
    X_sin_intercepto = modelo["X"][:, 1:]
    nombres = modelo["independientes"]
    n, k = X_sin_intercepto.shape
    filas = []
    for j, nombre in enumerate(nombres):
        if k == 1:
            r2_aux = 0.0
            vif = 1.0
        else:
            otras = np.delete(X_sin_intercepto, j, axis=1)
            X_aux = np.column_stack([np.ones(n), otras])
            if n <= X_aux.shape[1]:
                r2_aux = float("nan")
                vif = float("nan")
            else:
                r2_aux = _r2_auxiliar(X_aux, X_sin_intercepto[:, j])
                if r2_aux >= 1.0 - 1e-12:
                    vif = float("inf")
                else:
                    vif = 1.0 / (1.0 - r2_aux)
        filas.append({
            "Variable": nombre,
            "R² auxiliar": r2_aux,
            "VIF": vif,
            "Alerta": "VIF > 10" if np.isfinite(vif) and vif > 10 else
                      ("VIF infinito" if np.isinf(vif) else "OK"),
        })
    return pd.DataFrame(filas).set_index("Variable")


def ejecutar_pruebas(modelo: dict) -> dict:
    """Agrupa todas las pruebas de validación del modelo."""
    return {
        "normalidad": prueba_normalidad(modelo["residuos"]),
        "durbin_watson": {
            "estadistico": modelo["durbin_watson"],
            "nivel": interpretar_durbin_watson(modelo["durbin_watson"])[0],
            "conclusion": interpretar_durbin_watson(modelo["durbin_watson"])[1],
        },
        "breusch_pagan": prueba_breusch_pagan(modelo),
        "white": prueba_white(modelo),
        "vif": calcular_vif(modelo),
    }
