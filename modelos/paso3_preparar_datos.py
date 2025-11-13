import pandas as pd

def cargar_y_preparar(ruta_excel):
    # -------------------------------
    # 1. LEER EXCEL
    # -------------------------------
    df = pd.read_excel(ruta_excel)

    # Asegurar que columnas existan
    if "fecha" not in df.columns or "ventas" not in df.columns:
        raise ValueError("El Excel debe tener columnas: fecha y ventas")

    # -------------------------------
    # 2. FORMATEAR FECHA
    # -------------------------------
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])  # eliminar fechas inválidas

    # -------------------------------
    # 3. ORDENAR
    # -------------------------------
    df = df.sort_values("fecha")

    # -------------------------------
    # 4. Reset index
    # -------------------------------
    df = df.reset_index(drop=True)

    # -------------------------------
    # 5. Mostrar info
    # -------------------------------
    print("Datos cargados correctamente:")
    print(df.head())
    print(df.tail())

    # -------------------------------
    # 6. Retornar df limpio
    # -------------------------------
    return df


# --------------------------------------------------
# PRUEBA DEL SISTEMA (no borrar)
# --------------------------------------------------
if __name__ == "__main__":
    df = cargar_y_preparar("../data/ventas_producto.xlsx")
    print("\nFilas totales:", len(df))
