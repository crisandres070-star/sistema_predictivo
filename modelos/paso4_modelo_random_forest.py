import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def entrenar_random_forest(df):

    df = df.copy()

    # Convertir fechas a número para el modelo
    df["dia_n"] = df["fecha"].map(pd.Timestamp.toordinal)

    X = df[["dia_n"]]
    y = df["ventas"]

    modelo = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    modelo.fit(X, y)

    print("✔ RandomForest entrenado correctamente.")
    return modelo


def predecir_random_forest(modelo, df, dias_futuro=30):

    ultimo_dia = df["fecha"].max().toordinal()

    dias_futuros = np.array([ultimo_dia + i for i in range(1, dias_futuro + 1)]).reshape(-1, 1)

    predicciones = modelo.predict(dias_futuros)

    fechas_futuras = pd.date_range(start=df["fecha"].max() + pd.Timedelta(days=1), periods=dias_futuro)

    resultado = pd.DataFrame({
        "fecha": fechas_futuras,
        "prediccion": predicciones
    })

    print("✔ Predicción Random Forest completada.")
    return resultado


# --------------- TEST --------------------
if __name__ == "__main__":
    from modelos.paso3_preparar_datos import cargar_y_preparar

    df = cargar_y_preparar("../data/ventas_producto.xlsx")
    modelo = entrenar_random_forest(df)
    pred = predecir_random_forest(modelo, df, dias_futuro=30)

    print(pred)
