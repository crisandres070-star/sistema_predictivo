import pandas as pd
import matplotlib.pyplot as plt
from paso3_preparar_datos import cargar_y_preparar
from paso3_motor_inteligente import elegir_modelo
from paso4_modelo_random_forest import entrenar_random_forest, predecir_random_forest
from paso4_modelo_prophet import entrenar_prophet, predecir_prophet


def generar_grafico(df, pred):

    plt.figure(figsize=(14, 7))

    # Historial
    plt.plot(df["fecha"], df["ventas"], label="Histórico", color="#4c8bf5", linewidth=2)

    # Predicción
    plt.plot(pred["fecha"], pred["prediccion"], label="Predicción 30 días", color="#f5426f", linewidth=2)

    # Punto final
    plt.scatter(pred["fecha"].iloc[0], pred["prediccion"].iloc[0], color="#f5426f")

    # Estilo
    plt.title("Predicción automática de demanda", fontsize=16, fontweight="bold")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    
    plt.savefig("../resultado_prediccion.png")
    plt.close()

    print("✔ Gráfico guardado como resultado_prediccion.png")


def sistema_autonomo():

    print("\n📌 SISTEMA AUTÓNOMO INICIADO")
    print("Leyendo archivo...")

    df = cargar_y_preparar("../data/ventas_producto.xlsx")

    print("\n📌 Seleccionando modelo automáticamente...")
    modelo_usado = elegir_modelo(df)

    if modelo_usado == "random_forest":
        print("\n📌 Entrenando RANDOM FOREST...")
        modelo = entrenar_random_forest(df)
        pred = predecir_random_forest(modelo, df, 30)

    else:
        print("\n📌 Entrenando PROPHET...")
        modelo = entrenar_prophet(df)
        pred = predecir_prophet(modelo, 30)

    print("\n📌 Exportando predicciones...")
    pred.to_excel("../predicciones_30_dias.xlsx", index=False)

    print("✔ Archivo generado: predicciones_30_dias.xlsx")

    print("\n📌 Generando gráfico profesional...")
    generar_grafico(df, pred)

    print("\n🎉 PROCESO COMPLETO. SISTEMA AUTÓNOMO LISTO.")


if __name__ == "__main__":
    sistema_autonomo()
