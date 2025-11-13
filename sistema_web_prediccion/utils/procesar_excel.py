import pandas as pd

def cargar_y_preparar(df_excel):
    df = df_excel.copy()
    df.columns = ["fecha", "ventas"]

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna()

    df = df.sort_values("fecha")
    df = df.reset_index(drop=True)

    return df
