"""
test_regresion.py
Pruebas unitarias del núcleo estadístico del sistema de regresión.

Ejecutar desde la raíz del proyecto:
    python -m unittest pruebas/test_regresion.py -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from regresion import (  # noqa: E402
    ajustar_regresion,
    calcular_vif,
    detectar_delimitador,
    durbin_watson,
    ejecutar_pruebas,
    estadisticas_descriptivas,
    leer_csv_desde_bytes,
    limpiar_datos,
    p_valor_chi2,
    p_valor_f,
    p_valor_t,
    prueba_normalidad,
    proceso_estimacion,
)


class TestInferencia(unittest.TestCase):
    def test_p_valor_t_conocido(self):
        # t = 2.0 con 10 grados de libertad -> p bilateral ≈ 0.0734.
        self.assertAlmostEqual(p_valor_t(2.0, 10), 0.073388, places=5)

    def test_p_valor_chi2_conocido(self):
        # El percentil 95 de chi-cuadrado con 1 gl es 3.841.
        self.assertAlmostEqual(p_valor_chi2(3.841, 1), 0.05, places=3)

    def test_p_valor_f_conocido(self):
        self.assertAlmostEqual(p_valor_f(4.0, 1, 100), 0.0482, places=3)


class TestModelo(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        n = 80
        x1 = rng.normal(10, 2, n)
        x2 = rng.normal(5, 1.5, n)
        x3 = rng.uniform(20, 40, n)
        ruido = rng.normal(0, 1.0, n)
        y = 4.0 + 2.5 * x1 - 1.2 * x2 + 0.35 * x3 + ruido
        self.df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    def test_recupera_coeficientes(self):
        modelo = ajustar_regresion(self.df, "y", ["x1", "x2", "x3"])
        self.assertAlmostEqual(modelo["beta"][0], 4.0, delta=0.5)
        self.assertAlmostEqual(modelo["beta"][1], 2.5, delta=0.15)
        self.assertAlmostEqual(modelo["beta"][2], -1.2, delta=0.2)
        self.assertAlmostEqual(modelo["beta"][3], 0.35, delta=0.1)

    def test_estadisticas_del_modelo(self):
        modelo = ajustar_regresion(self.df, "y", ["x1", "x2", "x3"])
        self.assertGreater(modelo["r2"], 0.90)
        self.assertGreater(modelo["f"], 0)
        self.assertTrue(0 <= modelo["p_f"] <= 1)
        self.assertEqual(modelo["n"], 80)
        self.assertEqual(modelo["gl_residual"], 76)
        self.assertTrue(1.0 < modelo["durbin_watson"] < 3.0)

    def test_tabla_de_coeficientes_completa(self):
        modelo = ajustar_regresion(self.df, "y", ["x1", "x2"])
        tabla = modelo["coeficientes"]
        self.assertEqual(list(tabla["Variable"]), ["Intercepto", "x1", "x2"])
        self.assertTrue(np.isfinite(tabla["Coeficiente"]).all())
        self.assertTrue(np.isfinite(tabla["Error estándar"]).all())
        self.assertTrue(np.isfinite(tabla["Estadístico t"]).all())
        self.assertTrue(np.isfinite(tabla["p-valor"]).all())

    def test_vif_bajo_con_predictores_independientes(self):
        modelo = ajustar_regresion(self.df, "y", ["x1", "x2", "x3"])
        vif = calcular_vif(modelo)
        self.assertTrue((vif["VIF"] < 5).all())

    def test_vif_detecta_colinealidad(self):
        df = self.df.copy()
        df["x4"] = df["x1"] * 2.0 + 0.01 * np.random.default_rng(4).normal(size=len(df))
        modelo = ajustar_regresion(df, "y", ["x1", "x2", "x3", "x4"])
        vif = calcular_vif(modelo)
        self.assertGreater(vif.loc["x1", "VIF"], 10)
        self.assertGreater(vif.loc["x4", "VIF"], 10)

    def test_ecuacion_no_falla(self):
        from regresion import ecuacion_modelo

        modelo = ajustar_regresion(self.df, "y", ["x1", "x2"])
        texto = ecuacion_modelo(modelo)
        self.assertIn("y =", texto)
        self.assertIn("x1", texto)
        self.assertIn("x2", texto)

    def test_proceso_estimacion_reproduce_coeficientes(self):
        modelo = ajustar_regresion(self.df, "y", ["x1", "x2", "x3"])
        proceso = proceso_estimacion(modelo)
        np.testing.assert_allclose(proceso["beta"], modelo["beta"], rtol=1e-8, atol=1e-8)
        self.assertLess(proceso["error_maximo_verificacion"], 1e-8)
        self.assertEqual(proceso["XtX"].shape, (4, 4))
        self.assertEqual(proceso["Xty"].shape, (4,))
        self.assertEqual(proceso["metodo_inversa"], "inversa")


class TestPruebasDiagnosticas(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        x1 = rng.normal(0, 1, 120)
        x2 = rng.normal(0, 1, 120)
        y = 3 + 1.5 * x1 - 0.7 * x2 + rng.normal(0, 1, 120)
        self.df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        self.modelo = ajustar_regresion(self.df, "y", ["x1", "x2"])

    def test_normalidad_jarque_bera(self):
        resultado = prueba_normalidad(self.modelo["residuos"])
        self.assertTrue(resultado["aplicable"])
        self.assertTrue(0 <= resultado["p_jb"] <= 1)
        self.assertTrue(0 <= resultado["p_ks"] <= 1)
        self.assertFalse(resultado["rechaza_jb"])

    def test_durbin_watson_cerca_de_dos(self):
        dw = durbin_watson(self.modelo["residuos"])
        self.assertTrue(1.5 < dw < 2.5)

    def test_ejecutar_pruebas_agrupa_resultados(self):
        pruebas = ejecutar_pruebas(self.modelo)
        self.assertIn("normalidad", pruebas)
        self.assertIn("durbin_watson", pruebas)
        self.assertIn("breusch_pagan", pruebas)
        self.assertIn("white", pruebas)
        self.assertIn("vif", pruebas)
        self.assertTrue(pruebas["breusch_pagan"]["aplicable"])
        self.assertTrue(pruebas["white"]["aplicable"])
        self.assertTrue(0 <= pruebas["breusch_pagan"]["p"] <= 1)
        self.assertTrue(0 <= pruebas["white"]["p"] <= 1)


class TestDatos(unittest.TestCase):
    def test_detectar_delimitador_coma(self):
        self.assertEqual(detectar_delimitador("a,b,c\n1,2,3\n4,5,6"), ",")

    def test_detectar_delimitador_punto_y_coma(self):
        self.assertEqual(detectar_delimitador("a;b;c\n1;2;3\n4;5;6"), ";")

    def test_leer_csv_desde_bytes(self):
        contenido = "y;x1;x2\n1;2;3\n4;5;6\n".encode("utf-8")
        df, delimitador, codificacion = leer_csv_desde_bytes(contenido)
        self.assertEqual(delimitador, ";")
        self.assertEqual(codificacion, "utf-8")
        self.assertEqual(list(df.columns), ["y", "x1", "x2"])
        self.assertEqual(df.shape, (2, 3))

    def test_limpiar_datos_imputa_mediana(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [2.0, 4.0, 6.0]})
        limpio, reporte = limpiar_datos(df, "Imputar con la mediana", False)
        self.assertEqual(reporte["faltantes_finales"], 0)
        self.assertFalse(limpio["a"].isna().any())
        self.assertAlmostEqual(limpio.loc[2, "a"], 1.5)

    def test_limpiar_datos_elimina_filas_incompletas(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
        limpio, reporte = limpiar_datos(df, "Eliminar filas con faltantes", False)
        self.assertEqual(len(limpio), 2)
        self.assertEqual(reporte["filas_eliminadas"], 1)
        self.assertEqual(reporte["faltantes_finales"], 0)

    def test_estadisticas_descriptivas(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        tabla = estadisticas_descriptivas(df)
        self.assertAlmostEqual(tabla.loc["x", "Media"], 2.5)
        self.assertAlmostEqual(tabla.loc["x", "Varianza"], 1.6666667, places=5)
        self.assertAlmostEqual(tabla.loc["x", "Mediana"], 2.5)
        self.assertAlmostEqual(tabla.loc["x", "Q1"], 1.75)


if __name__ == "__main__":
    unittest.main()
