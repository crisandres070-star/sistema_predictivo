import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------
# 1. Elegir modelo automáticamente según días de datos
# ---------------------------------------------------
def elegir_modelo(df):
    dias = df["fecha"].nunique()
    print(f"📌 Días detectados: {dias}")

    if dias < 100:
        print("🟢 Usando Random Forest (pocos datos)")
        return "random_forest"
    else:
        print("🔵 Usando Prophet (datos suficientes)")
        return "prophet"


# ---------------------------------------------------
# 2. Entrenar Random Forest
# ---------------------------------------------------
def entrenar_rf(df):
    df = df.copy()
    df["fecha_ordinal"] = df["fecha"].map(pd.Timestamp.toordinal)

    X = df[["fecha_ordinal"]]
    y = df["ventas"]

    modelo = RandomForestRegressor(n_estimators=200)
    modelo.fit(X, y)

    return modelo


# ---------------------------------------------------
# 3. Predecir con RF
# ---------------------------------------------------
def predecir_rf(modelo, df, dias=30):
    ultima_fecha = df["fecha"].max()
    fechas_futuras = [ultima_fecha + pd.Timedelta(days=i) for i in range(1, dias+1)]

    fechas_ord = [f.toordinal() for f in fechas_futuras]
    pred = modelo.predict(pd.DataFrame({"fecha_ordinal": fechas_ord}))

    return pd.DataFrame({"fecha": fechas_futuras, "prediccion": pred})


# ---------------------------------------------------
# 4. Entrenar PROPHET
# ---------------------------------------------------
def entrenar_prophet(df):
    df_p = df.rename(columns={"fecha": "ds", "ventas": "y"})
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    modelo.fit(df_p)
    return modelo


# ---------------------------------------------------
# 5. Predecir con PROPHET
# ---------------------------------------------------
def predecir_prophet(modelo, dias=30):
    futuro = modelo.make_future_dataframe(periods=dias)
    forecast = modelo.predict(futuro)
    pred = forecast.tail(dias)[["ds", "yhat"]]
    return pred.rename(columns={"ds": "fecha", "yhat": "prediccion"})
