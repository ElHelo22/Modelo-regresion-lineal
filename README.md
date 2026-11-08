# 📊 Sistema de Regresión Lineal Multivariada

Aplicación desarrollada para la práctica de **Investigación Operativa II**. El sistema permite cargar un archivo CSV, limpiar los datos, estimar un modelo de regresión lineal multivariada por mínimos cuadrados ordinarios (MCO), validarlo con pruebas estadísticas y generar los gráficos solicitados en la guía de práctica.

La interfaz web está desarrollada con **Streamlit**, y el núcleo estadístico se implementó con **NumPy y Pandas**, sin depender de `statsmodels` ni `scipy`.

---

## 👥 Integrantes

* **Apaza Mamani Helmer Rudel**
* **Callizaya Lopez Naeli Daniela**
* **Chavez Mamani Joel Alexix**
* **Mamani Arias Karen Belen**
* **Mamani Quispe Miguel Angel**
* **Mendoza Paye Cristian Rodrigo**

---

## 🛠️ Tecnologías utilizadas

* 🐍 **Python 3**
* 🎈 **Streamlit** — interfaz web interactiva
* 🐼 **Pandas** — carga, limpieza y análisis de datos
* 🔢 **NumPy** — cálculo matricial y estimación del modelo
* 📈 **Matplotlib** — generación de gráficos
* 📄 **CSV** — formato de entrada de datos

---

## 📋 Requisitos

* Python 3.10 o superior
* `pip`
* Conexión a Internet para instalar las dependencias

Para comprobar la instalación:

```bash
python --version
python -m pip --version
```

En Windows también puede usarse `py` en lugar de `python`.

---

## 📥 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ElHelo22/proyecto-io.git
cd proyecto-io
```

### 2. Actualizar pip (recomendado)

```bash
python -m pip install --upgrade pip
```

### 3. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Ejecución

Desde la raíz del proyecto:

```bash
python -m streamlit run app/main.py
```

Streamlit mostrará una dirección local similar a:

```text
http://localhost:8501
```

---

## 📁 Estructura del proyecto

```text
Modelo-regresion-lineal/
│
├── app/
│   ├── main.py                 # Interfaz web Streamlit
│   ├── regresion.py            # Carga, limpieza, MCO y pruebas estadísticas
│   ├── graficos_regresion.py   # Gráficos del modelo
│   └── analisis_ejemplo.py     # Ejecución reproducible por consola
│
├── datos/
│   └── regresion.csv           # Datos de ejemplo para el modelo
│
├── resultados/                 # Reporte y gráficos generados por analisis_ejemplo.py
│
├── pruebas/
│   └── test_regresion.py       # Pruebas unitarias del sistema de regresión
│
├── requirements.txt
├── guia_practica_io_ii.md
└── README.md
```

---

## 📂 Datos de entrada

El archivo CSV debe contener una columna para la variable dependiente **Y** y una o más columnas numéricas para las variables independientes **X**.

El sistema acepta delimitadores:

* coma (`,`)
* punto y coma (`;`)
* tabulación (`\t`)
* barra vertical (`|`)

La detección del delimitador es automática. También se muestra una vista previa y un resumen de tipos de datos y valores faltantes.

### Datos de ejemplo

El archivo `datos/regresion.csv` contiene 80 observaciones simuladas con las variables:

| Variable | Descripción |
| --- | --- |
| `ventas_miles` | Ventas mensuales en miles de bolivianos (variable dependiente) |
| `publicidad_miles` | Gasto mensual en publicidad, en miles de bolivianos |
| `precio_promedio` | Precio promedio del producto |
| `clientes_mensuales` | Número de clientes atendidos por mes |
| `superficie_m2` | Superficie del establecimiento en metros cuadrados |

El modelo de ejemplo estimado es:

```text
ventas_miles = 8.3529 + 1.8494·publicidad_miles − 0.8029·precio_promedio
               + 0.0570·clientes_mensuales + 0.0215·superficie_m2
```

---

## ✅ Funcionalidades implementadas

### 1. Carga y limpieza de datos

* Carga de archivos CSV desde la interfaz.
* Detección automática del delimitador y de la codificación.
* Vista previa de las primeras filas.
* Validación y conversión de variables numéricas.
* Tratamiento de valores faltantes:
  * eliminación de filas incompletas;
  * imputación con la media;
  * imputación con la mediana;
  * conservación de faltantes.
* Detección de valores atípicos mediante la regla **1.5 × IQR**.
* Opción para eliminar los valores atípicos antes de estimar el modelo.
* Estadísticas descriptivas: media, varianza, desviación estándar, mínimo, cuartiles, mediana, máximo y faltantes.

### 2. Estimación del modelo

Se estima el modelo:

\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_k X_k + \varepsilon
\]

mediante mínimos cuadrados ordinarios. La interfaz muestra:

* Coeficientes estimados \(\beta_j\).
* Error estándar de cada coeficiente.
* Estadístico \(t\).
* Valor \(p\) bilateral.
* Nivel de significancia (`***`, `**`, `*`, `ns`).
* \(R^2\), \(R^2\) ajustado, estadístico \(F\), \(p\)-valor global, error estándar de la regresión y Durbin-Watson.
* **Proceso de estimación paso a paso:** modelo propuesto, función objetivo, matriz de diseño \(\mathbf{X}\), vector \(\mathbf{Y}\), productos \(\mathbf{X}^T\mathbf{X}\) y \(\mathbf{X}^T\mathbf{Y}\), inversa de \(\mathbf{X}^T\mathbf{X}\), solución \(\hat{\beta}\), verificación de las ecuaciones normales, valores ajustados, residuos y ecuación final.

### 3. Validación del modelo

* **Normalidad de los residuos:**
  * Jarque-Bera (prueba principal).
  * Kolmogorov-Smirnov con parámetros estimados (aproximada).
  * Conclusión automática con \(\alpha = 0.05\).
* **Autocorrelación:**
  * Estadístico Durbin-Watson.
  * Interpretación de autocorrelación positiva, negativa o ausencia de evidencia.
* **Homocedasticidad:**
  * Prueba de Breusch-Pagan.
  * Prueba de White.
  * Conclusión automática.
* **Multicolinealidad:**
  * VIF para cada variable independiente.
  * Resaltado en rojo de los coeficientes con VIF mayor que 10.

### 4. Gráficos

* Diagrama de dispersión de cada \(X_i\) frente a \(Y\), con recta de tendencia y correlación.
* Histograma de residuos con curva normal de referencia.
* Gráfico Q-Q normal de los residuos.
* Residuos estandarizados contra valores ajustados.
* Matriz de correlaciones entre las variables del modelo.
* Descarga individual de cada gráfico en formato PNG.

### 5. Reporte

* Generación de un reporte de texto con datos, modelo, coeficientes, pruebas y VIF.
* Descarga del reporte en formato `.txt`.

---

## 🧪 Pruebas unitarias

El núcleo estadístico incluye pruebas de regresión, inferencia, VIF, pruebas diagnósticas y limpieza de datos. Para ejecutarlas:

```bash
python -m unittest pruebas/test_regresion.py -v
```

---

## 🖥️ Análisis reproducible por consola

Además de la interfaz web, el proyecto incluye un script que ejecuta el modelo sobre un CSV y guarda el reporte y todos los gráficos en `resultados/`:

```bash
python app/analisis_ejemplo.py
```

También puede aplicarse a otro archivo indicando la variable dependiente y las independientes:

```bash
python app/analisis_ejemplo.py --csv mi_archivo.csv --y Y --x X1 X2 X3
```

Los archivos generados son:

* `resultados/reporte_regresion.txt`
* `resultados/dispersion_*.png`
* `resultados/histograma_residuos.png`
* `resultados/qq_residuos.png`
* `resultados/residuos_estandarizados.png`
* `resultados/matriz_correlaciones.png`

---

## ⚠️ Solución de problemas

### `ModuleNotFoundError`

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

### `streamlit no se reconoce como un comando`

Ejecutar Streamlit como módulo de Python:

```bash
python -m streamlit run app/main.py
```

### El archivo CSV no se lee correctamente

Verificar que:

* el archivo tenga una fila de encabezados;
* las variables del modelo sean numéricas o convertibles a número;
* el delimitador sea coma, punto y coma, tabulación o barra vertical.

Si el problema persiste, guardar el archivo con codificación UTF-8.

---

## 🎓 Proyecto académico

**Materia:** Investigación Operativa II<br>
**Docente:** M.Sc. Ing. Juan Carlos Catunta Choquecalle<br>
**Práctica:** Sistema de Regresión Multivariada<br>
**Gestión:** 2026

---

## 📄 Licencia

Proyecto desarrollado con fines académicos y educativos.
