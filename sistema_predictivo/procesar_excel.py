import pandas as pd


def cargar_excel(archivo):
    df = pd.read_excel(archivo)
    
    # Normalizar columnas
    df.columns = df.columns.str.lower()

    if "fecha" not in df.columns or "ventas" not in df.columns:
        raise ValueError("El Excel debe tener columnas: fecha, ventas")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df.dropna(subset=["fecha", "ventas"], inplace=True)

    # Crear índice numérico
    df["dia"] = range(1, len(df) + 1)

    return df
