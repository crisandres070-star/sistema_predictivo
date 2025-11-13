from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def elegir_modelo(df):
    return "random_forest"


def entrenar_random_forest(df):
    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo.fit(df[["dia"]], df["ventas"])
    return modelo


def predecir_random_forest(modelo, df, dias_futuro=30):
    ult_dia = df["dia"].max()
    dias_futuros = list(range(ult_dia + 1, ult_dia + dias_futuro + 1))

    predicciones = modelo.predict([[d] for d in dias_futuros])

    resultado = pd.DataFrame({
        "fecha": pd.date_range(df["fecha"].max(), periods=dias_futuro+1, freq="D")[1:],
        "prediccion": predicciones
    })

    return resultado
