"""Carga de los archivos crudos y limpieza de sales_train.

Reproduce las decisiones de limpieza del notebook 01 (EDA) para que todos los
notebooks del Proyecto 2 partan exactamente de los mismos datos. La unificación
de tiendas duplicadas NO se hace aquí: la hace el operador Mapper
(ver predict_future_sales.transformers y predict_future_sales.features).
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_DIR / "data" / "raw"

# Tipos pequeños para ahorrar memoria (shop_id < 60, item_id < 22,170, date_block_num < 35)
SALES_DTYPES = {
    "date": "category",
    "date_block_num": "int8",
    "shop_id": "int8",
    "item_id": "int16",
    "item_price": "float32",
    "item_cnt_day": "float32",
}

# Umbrales de outliers definidos en la sección 7.c del EDA
MAX_ITEM_PRICE = 100_000
MAX_ITEM_CNT_DAY = 1_000


def load_raw(data_dir=DATA_RAW):
    """Lee los CSV crudos con tipos de dato reducidos y devuelve un diccionario de DataFrames."""
    data_dir = Path(data_dir)
    return {
        "sales": pd.read_csv(data_dir / "sales_train.csv", dtype=SALES_DTYPES),
        "items": pd.read_csv(
            data_dir / "items.csv", dtype={"item_id": "int16", "item_category_id": "int8"}
        ),
        "item_categories": pd.read_csv(
            data_dir / "item_categories.csv", dtype={"item_category_id": "int8"}
        ),
        "shops": pd.read_csv(data_dir / "shops.csv", dtype={"shop_id": "int8"}),
        "test": pd.read_csv(
            data_dir / "test.csv", dtype={"ID": "int32", "shop_id": "int8", "item_id": "int16"}
        ),
    }


def clean_sales(sales):
    """Aplica las decisiones de limpieza del EDA (secciones 4.b, 4.c y 7.c).

    1. Imputa los precios <= 0 con la mediana del mismo producto en el mismo mes.
    2. Elimina los duplicados exactos (se conserva la primera aparición).
    3. Elimina las transacciones extremas: item_price > 100,000 o item_cnt_day >= 1,000.
    """
    sales = sales.copy()

    # 1. Precio inválido
    invalidos = sales["item_price"] <= 0
    if invalidos.any():
        validos = sales[~invalidos]
        mediana = validos.groupby(["item_id", "date_block_num"], observed=True)[
            "item_price"
        ].median()
        llaves = pd.MultiIndex.from_frame(sales.loc[invalidos, ["item_id", "date_block_num"]])
        sales.loc[invalidos, "item_price"] = mediana.reindex(llaves).to_numpy()

    # 2. Duplicados exactos
    sales = sales.drop_duplicates(keep="first")

    # 3. Outliers extremos
    outliers = (sales["item_price"] > MAX_ITEM_PRICE) | (sales["item_cnt_day"] >= MAX_ITEM_CNT_DAY)
    sales = sales[~outliers]

    return sales.drop(columns="date").reset_index(drop=True)
