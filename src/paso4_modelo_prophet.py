import pandas as pd
from prophet import Prophet

def entrenar_prophet(df):
    df_p = df.rename(columns={"fecha": "ds", "ventas": "y"})

    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )

    modelo.fit(df_p)
    print("✔ Modelo Prophet entrenado correctamente.")
    return modelo


def predecir_prophet(modelo, dias_futuro=30):
    futuro = modelo.make_future_dataframe(periods=dias_futuro)
    forecast = modelo.predict(futuro)

    pred = forecast[["ds", "yhat"]].tail(dias_futuro).rename(columns={"ds": "fecha", "yhat": "prediccion"})
    print("✔ Predicción Prophet completada.")
    return pred


# ---------------- TEST ---------------------
if __name__ == "__main__":
    from paso3_preparar_datos import cargar_y_preparar

    df = cargar_y_preparar("../data/ventas_producto.xlsx")
    modelo = entrenar_prophet(df)
    pred = predecir_prophet(modelo, dias_futuro=30)

    print(pred)
