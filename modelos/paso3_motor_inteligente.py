from modelos.paso3_preparar_datos import cargar_y_preparar

def elegir_modelo(df):
    # Cantidad de días únicos
    dias = df["fecha"].nunique()

    print(f"El archivo contiene {dias} días de datos.")

    if dias < 90:
        print("➡ Usando MODELO: RandomForestRegressor")
        return "random_forest"
    else:
        print("➡ Usando MODELO: Prophet (Meta)")
        return "prophet"


# -----------------------
# PRUEBA (NO BORRAR)
# -----------------------
if __name__ == "__main__":
    df = cargar_y_preparar("../data/ventas_producto.xlsx")
    modelo = elegir_modelo(df)
    print("\nModelo seleccionado automáticamente:", modelo)
