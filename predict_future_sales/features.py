"""Construcción de la tabla de modelado mensual (tienda x producto) con sus lags.

Los lags NO forman parte del pipeline de sklearn: para calcularlos se necesita
el historial de ventas completo y no solo la fila que se va a predecir. Por eso
se construyen con `build_feature_matrix`, que se usa igual en los notebooks
02, 03, 04 y 05. Así, las filas de entrenamiento, validación y test (mes 34)
se generan con exactamente el mismo código.

Regla contra la fuga de información: toda variable del mes m se calcula solo
con datos de los meses < m (los lags del mes 34 salen de los meses 33, 32, ...).
"""

import numpy as np
import pandas as pd

from predict_future_sales.dataset import clean_sales
from predict_future_sales.transformers import Mapper

# --- Constantes compartidas por los notebooks -------------------------------------------------

TARGET = "item_cnt_month"
TARGET_CLIP = (0, 20)  # mismo recorte que aplica la competencia de Kaggle al evaluar
TEST_MONTH = 34
FIRST_MODEL_MONTH = 12  # primer mes con lag de 12 meses disponible
LAST_TRAIN_MONTH = 26  # división secuencial 80/20: entrenamiento 0-26, validación 27-33

# Tiendas duplicadas detectadas en el EDA (sección 4.c): se unifican hacia el ID de test
SHOP_MAPPINGS = {0: 57, 1: 58, 11: 10}

# Lags del par tienda-producto elegidos en el notebook 02
LAGS = [1, 2, 3, 6, 12]

# Prefijos de shop_name que no son una ciudad (tiendas sin ubicación física)
NON_CITY_PREFIXES = {"Выездная", "Интернет-магазин", "Цифровой"}

KEYS = ["date_block_num", "shop_id", "item_id"]
CATEGORICAL_FEATURES = ["shop_city", "item_cat_type", "item_cat_subtype"]
EXOGENOUS_ID_FEATURES = ["shop_id", "item_category_id"]
MEAN_LAG_FEATURES = ["item_avg_cnt_lag_1", "shop_avg_cnt_lag_1", "shop_cat_avg_cnt_lag_1"]
PRICE_FEATURES = ["item_price_last"]


def pair_lag_features(lags=LAGS):
    return [f"{TARGET}_lag_{lag}" for lag in lags]


def count_lag_features(lags=LAGS):
    """Variables de conteo rezagadas: un NaN significa que no hubo ventas en ese mes."""
    return pair_lag_features(lags) + MEAN_LAG_FEATURES


# --- Variables estáticas de los catálogos ------------------------------------------------------


def shop_features(shops):
    """Ciudad extraída del nombre de la tienda; NaN para las tiendas sin ubicación física."""
    primera = shops["shop_name"].str.split(" ").str[0].str.replace("!", "", regex=False)
    ciudad = primera.where(~primera.isin(NON_CITY_PREFIXES))
    return pd.DataFrame({"shop_id": shops["shop_id"], "shop_city": ciudad.astype("category")})


def category_features(item_categories):
    """Tipo (antes de ' - ') y subtipo (después de ' - ') del nombre de la categoría.

    Las categorías sin ' - ' (por ejemplo 'Служебные' o 'Доставка товара') no tienen
    subtipo, por lo que item_cat_subtype queda como NaN.
    """
    nombre = item_categories["item_category_name"]
    partes = nombre.str.split(" - ", n=1)
    tipo = partes.str[0].str.replace(r"\s*\(.*\)$", "", regex=True).str.strip()
    subtipo = partes.str[1].str.strip()
    return pd.DataFrame(
        {
            "item_category_id": item_categories["item_category_id"],
            "item_cat_type": tipo.astype("category"),
            "item_cat_subtype": subtipo.astype("category"),
        }
    )


# --- Construcción de la tabla mensual ----------------------------------------------------------


def build_grid(sales, test):
    """Cuadrícula mensual: en cada mes, todas las tiendas activas x todos los productos vendidos.

    Se agrega el mes 34 con los pares de test.csv.
    """
    bloques = []
    for mes, g in sales.groupby("date_block_num"):
        tiendas = g["shop_id"].unique()
        productos = g["item_id"].unique()
        bloques.append(
            pd.DataFrame(
                {
                    "date_block_num": np.int8(mes),
                    "shop_id": np.repeat(tiendas, len(productos)).astype("int8"),
                    "item_id": np.tile(productos, len(tiendas)).astype("int16"),
                }
            )
        )
    bloques.append(
        pd.DataFrame(
            {
                "date_block_num": np.int8(TEST_MONTH),
                "shop_id": test["shop_id"].astype("int8").to_numpy(),
                "item_id": test["item_id"].astype("int16").to_numpy(),
            }
        )
    )
    return pd.concat(bloques, ignore_index=True)


def add_target(grid, sales):
    """Suma item_cnt_day por mes-tienda-producto, rellena con 0 y recorta a [0, 20]."""
    mensual = sales.groupby(KEYS, observed=True)["item_cnt_day"].sum().rename(TARGET).reset_index()
    grid = grid.merge(mensual, on=KEYS, how="left")
    historico = grid["date_block_num"] < TEST_MONTH
    grid.loc[historico, TARGET] = grid.loc[historico, TARGET].fillna(0)
    grid[TARGET] = grid[TARGET].clip(*TARGET_CLIP).astype("float32")  # el mes 34 queda NaN
    return grid


def add_static_features(grid, items, item_categories, shops):
    grid["month"] = (grid["date_block_num"] % 12 + 1).astype("int8")
    cat_por_item = items.set_index("item_id")["item_category_id"]
    grid["item_category_id"] = grid["item_id"].map(cat_por_item).astype("int8")
    grid = grid.merge(shop_features(shops), on="shop_id", how="left")
    grid = grid.merge(category_features(item_categories), on="item_category_id", how="left")
    return grid


def add_lags(grid, col, keys, lags, nombre=None):
    """Agrega `col` desplazada `lag` meses hacia adelante: la fila del mes m recibe el valor de m - lag."""
    nombre = nombre or col
    base = grid[["date_block_num", *keys, col]]
    for lag in lags:
        desplazado = base.copy()
        desplazado["date_block_num"] = (desplazado["date_block_num"] + lag).astype("int8")
        desplazado = desplazado.rename(columns={col: f"{nombre}_lag_{lag}"})
        grid = grid.merge(desplazado, on=["date_block_num", *keys], how="left")
    return grid


def add_mean_lag(grid, keys, nombre):
    """Promedio mensual del objetivo por `keys`, rezagado un mes (nivel del producto, tienda, etc.)."""
    media = (
        grid.groupby(["date_block_num", *keys], observed=True)[TARGET]
        .mean()
        .astype("float32")
        .rename(f"{nombre}_lag_1")
        .reset_index()
    )
    # Se desplaza la tabla agregada (una fila por mes y clave), no la cuadrícula completa
    media["date_block_num"] = (media["date_block_num"] + 1).astype("int8")
    return grid.merge(media, on=["date_block_num", *keys], how="left")


def add_price_feature(grid, sales):
    """Último precio promedio conocido del producto ANTES del mes m (NaN si nunca se vendió)."""
    precio = sales.groupby(["item_id", "date_block_num"], observed=True)["item_price"].mean()
    tabla = precio.unstack().reindex(columns=range(TEST_MONTH + 1))
    ultimo = tabla.ffill(axis=1).shift(1, axis=1)  # columna m = último precio de los meses < m
    ultimo = ultimo.stack().rename("item_price_last").astype("float32").reset_index()
    ultimo["date_block_num"] = ultimo["date_block_num"].astype("int8")
    ultimo["item_id"] = ultimo["item_id"].astype("int16")
    return grid.merge(ultimo, on=["item_id", "date_block_num"], how="left")


def add_age_features(grid, sales):
    """Antigüedad del producto e indicadores de producto nuevo y de par nuevo (sin ventas previas)."""
    primer_item = sales.groupby("item_id")["date_block_num"].min()
    primer = grid["item_id"].map(primer_item)
    mes = grid["date_block_num"]
    grid["item_age"] = (mes - primer).clip(lower=0).fillna(0).astype("int8")
    grid["item_is_new"] = (primer.isna() | (primer >= mes)).astype("int8")

    primer_par = (
        sales.groupby(["shop_id", "item_id"])["date_block_num"].min().rename("_primer_par")
    )
    grid = grid.merge(primer_par.reset_index(), on=["shop_id", "item_id"], how="left")
    grid["shop_item_is_new"] = (
        grid["_primer_par"].isna() | (grid["_primer_par"] >= grid["date_block_num"])
    ).astype("int8")
    return grid.drop(columns="_primer_par")


def build_feature_matrix(raw, lags=LAGS, first_month=FIRST_MODEL_MONTH):
    """Tabla de modelado completa a partir de los archivos crudos (salida de dataset.load_raw).

    Pasos: limpieza del EDA -> unificación de tiendas con Mapper -> cuadrícula mensual con
    ceros -> objetivo recortado -> variables de catálogo -> lags -> filtro de meses.
    Devuelve las filas de los meses `first_month` a 34 (el mes 34 son los pares de test).
    """
    mapper = Mapper(variables=["shop_id"], mappings=SHOP_MAPPINGS)
    sales = mapper.transform(clean_sales(raw["sales"]))
    sales["shop_id"] = sales["shop_id"].astype("int8")
    test = mapper.transform(raw["test"])

    grid = build_grid(sales, test)
    grid = add_target(grid, sales)
    grid = add_static_features(grid, raw["items"], raw["item_categories"], raw["shops"])
    grid = add_lags(grid, TARGET, ["shop_id", "item_id"], lags)
    grid = add_mean_lag(grid, ["item_id"], "item_avg_cnt")
    grid = add_mean_lag(grid, ["shop_id"], "shop_avg_cnt")
    grid = add_mean_lag(grid, ["shop_id", "item_category_id"], "shop_cat_avg_cnt")
    grid = add_price_feature(grid, sales)
    grid = add_age_features(grid, sales)

    grid = grid[grid["date_block_num"] >= first_month].reset_index(drop=True)
    return grid


def split_xy(matrix, months):
    """Separa X (sin el objetivo) e y para los meses indicados."""
    filas = matrix["date_block_num"].isin(months)
    X = matrix.loc[filas].drop(columns=TARGET)
    y = matrix.loc[filas, TARGET]
    return X, y
