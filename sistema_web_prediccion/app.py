import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modelos.motor import (
    elegir_modelo,
    entrenar_rf,
    entrenar_prophet,
    predecir_rf,
    predecir_prophet
)

from utils.procesar_excel import cargar_y_preparar


st.set_page_config(page_title="Sistema Predictivo", layout="wide")

st.title("🤖 Sistema de Predicción Automática de Demanda")
st.write("Sube tu archivo Excel y obtén predicciones automáticas para los próximos 30 días.")


# -------- CARGAR ARCHIVO ------------
archivo = st.file_uploader("Subir archivo Excel", type=["xlsx"])

if archivo:
    df_excel = pd.read_excel(archivo)
    df = cargar_y_preparar(df_excel)

    st.subheader("📄 Datos cargados")
    st.dataframe(df.head())

    # ---- Elegir modelo automáticamente ----
    modelo_elegido = elegir_modelo(df)

    if modelo_elegido == "random_forest":
        modelo = entrenar_rf(df)
        pred = predecir_rf(modelo, df, dias=30)
    else:
        modelo = entrenar_prophet(df)
        pred = predecir_prophet(modelo, dias=30)

    st.subheader("📊 Predicción 30 días")
    st.dataframe(pred)

    # ----- Gráfico -----
    st.subheader("📈 Gráfico de Predicción")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["fecha"], df["ventas"], label="Histórico")
    ax.plot(pred["fecha"], pred["prediccion"], label="Predicción 30 días", color="red")
    ax.legend()
    st.pyplot(fig)

    # ---- Descarga ----
    # Exportar CSV compatible con Excel LATAM (usa ; como separador)
csv_latam = pred.to_csv(index=False, sep=';', encoding='utf-8')

st.download_button(
    label="💾 Descargar predicciones",
    data=csv_latam,
    file_name="predicciones_30_dias.csv",
    mime="text/csv"
)

