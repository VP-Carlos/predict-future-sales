"""Operadores propios compatibles con los pipelines de sklearn.

Viven en este módulo (y no dentro de un notebook) para que los archivos .pkl
que los contienen se puedan cargar desde cualquier notebook o script.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Mapper(BaseEstimator, TransformerMixin):
    """Reemplaza los valores de las variables indicadas usando un diccionario."""

    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError("variables debe ser una lista")
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].replace(self.mappings)
        return X


class DowncastFloat32(BaseEstimator, TransformerMixin):
    """Convierte todas las columnas numéricas a float32 para reducir el uso de memoria.

    Se usa como último paso del preprocesamiento: StandardScaler devuelve float64,
    lo que duplica el tamaño de la matriz de entrenamiento sin aportar precisión útil.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)

    def __sklearn_is_fitted__(self):
        # No aprende nada en fit; sin esto, sklearn consideraría "no ajustado" un pipeline
        # que termina en este paso.
        return True
