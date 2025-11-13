import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modelos.motor import elegir_modelo, entrenar_random_forest, predecir_random_forest
from modelos.motor import entrenar_prophet, predecir_prophet
from utils.procesar_excel import cargar_excel

st.title("🤖 Sistema de Predicción Automática de Demanda")
st.write("Sube tu archivo Excel y obtén predicciones automáticas para los próximos 30 días.")

archivo = st.file_uploader("Subir archivo Excel", type=["xlsx"])

if archivo:
    df = cargar_excel(archivo)

    st.subheader("📊 Datos cargados")
    st.dataframe(df)

    # -------- Elegir modelo --------
    modelo_elegido = elegir_modelo(df)

    if modelo_elegido == "prophet":
        st.info("🔮 Usando modelo: Prophet (Meta)")
        modelo = entrenar_prophet(df)
        pred = predecir_prophet(modelo, dias_futuro=30)

    else:
        st.info("🌳 Usando modelo: Random Forest")
        modelo = entrenar_random_forest(df)
        pred = predecir_random_forest(modelo, df, dias_futuro=30)

    # -------- Mostrar predicción --------
    st.subheader("📈 Predicción 30 días")
    st.dataframe(pred)

    # -------- Gráfico --------
    st.subheader("📉 Gráfico de Predicción")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["fecha"], df["ventas"], label="Histórico")
    ax.plot(pred["fecha"], pred["prediccion"], label="Predicción 30 días", color="red")
    ax.legend()
    st.pyplot(fig)

    # -------- Descargar CSV --------
    csv_data = pred.to_csv(index=False, encoding="utf-8")

    st.download_button(
        label="📥 Descargar predicciones",
        data=csv_data,
        file_name="predicciones_30_dias.csv",
        mime="text/csv"
    )
