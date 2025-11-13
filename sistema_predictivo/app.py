import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# ---------------------------
# Función para cargar el Excel
# ---------------------------
def cargar_excel(archivo_subido):
    """
    Lee un archivo Excel subido por el usuario y devuelve un DataFrame
    con las columnas 'fecha' y 'valor' (demanda/ventas).
    """
    df = pd.read_excel(archivo_subido)

    # Intentar detectar columnas de fecha y valor
    columnas = df.columns.tolist()

    # Buscar columna de fecha
    col_fecha = None
    for nombre in columnas:
        nombre_lower = str(nombre).lower()
        if "fecha" in nombre_lower or "date" in nombre_lower:
            col_fecha = nombre
            break

    # Si no encuentra, usa la primera columna
    if col_fecha is None:
        col_fecha = columnas[0]

    # Buscar columna de valor
    col_valor = None
    for nombre in columnas:
        nombre_lower = str(nombre).lower()
        if any(pal in nombre_lower for pal in ["venta", "demand", "cantidad", "valor", "unidades"]):
            if nombre != col_fecha:
                col_valor = nombre
                break

    # Si no encuentra, usa la segunda columna
    if col_valor is None:
        if len(columnas) < 2:
            raise ValueError(
                "El archivo debe tener al menos 2 columnas (fecha y valor numérico)."
            )
        col_valor = columnas[1]

    df = df[[col_fecha, col_valor]].copy()
    df.columns = ["fecha", "valor"]

    # Convertir fecha
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    # Asegurar que valor es numérico
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["valor"])

    # Ordenar por fecha
    df = df.sort_values("fecha").reset_index(drop=True)

    return df


# ---------------------------
# Función para entrenar modelo
# ---------------------------
def entrenar_modelo(df):
    """
    Entrena un RandomForestRegressor básico usando el índice temporal como feature.
    """
    # Usamos el índice (0,1,2,...,N-1) como variable X
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["valor"].values

    modelo = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X, y)
    return modelo


# ---------------------------
# Función para predecir 30 días
# ---------------------------
def predecir_30_dias(df, modelo, dias_futuro=30):
    """
    Genera predicciones para los próximos 'dias_futuro' puntos.
    Asume datos diarios.
    """
    n = len(df)

    # Índices futuros
    X_futuro = np.arange(n, n + dias_futuro).reshape(-1, 1)
    pred = modelo.predict(X_futuro)

    # Fechas futuras (suponiendo frecuencia diaria)
    ultima_fecha = df["fecha"].max()
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1),
        periods=dias_futuro,
        freq="D"
    )

    df_pred = pd.DataFrame({
        "fecha": fechas_futuras,
        "prediccion": pred
    })

    return df_pred


# ---------------------------
# APP STREAMLIT
# ---------------------------
st.set_page_config(
    page_title="Sistema de Predicción Automática de Demanda",
    layout="wide"
)

st.title("📈 Sistema de Predicción Automática de Demanda")
st.write("Sube tu archivo Excel y obtén predicciones automáticas para los próximos 30 días.")

archivo = st.file_uploader(
    "Subir archivo Excel",
    type=["xlsx", "xls"]
)

if archivo is not None:
    try:
        # 1) Cargar datos
        df = cargar_excel(archivo)

        st.subheader("📂 Datos cargados")
        st.write(f"Registros totales: **{len(df)}**")
        st.dataframe(df.tail(20))

        # 2) Entrenar modelo
        modelo = entrenar_modelo(df)

        # 3) Predicción 30 días
        df_pred = predecir_30_dias(df, modelo, dias_futuro=30)

        st.subheader("📊 Predicción 30 días")
        st.dataframe(df_pred)

        # 4) Gráfico
        st.subheader("📉 Gráfico de Predicción")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["fecha"], df["valor"], label="Histórico", color="tab:blue")
        ax.plot(df_pred["fecha"], df_pred["prediccion"], label="Predicción 30 días", color="tab:red")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # 5) Descargar CSV
        st.subheader("⬇️ Descargar predicciones")

        csv_bytes = df_pred.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Descargar predicciones (CSV)",
            data=csv_bytes,
            file_name="predicciones_30_dias.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Ocurrió un error al procesar el archivo: {e}")
else:
    st.info("Sube un archivo Excel para comenzar.")
