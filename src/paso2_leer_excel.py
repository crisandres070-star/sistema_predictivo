import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("../data/ventas_producto.xlsx")

# Convertir fecha (modo tolerante)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.dropna(subset=["fecha"])

# Ordenar
df = df.sort_values("fecha")

print("\nPRIMERAS FILAS DEL EXCEL:")
print(df.head())

print("\nÚLTIMAS FILAS:")
print(df.tail())

plt.figure(figsize=(12,5))
plt.plot(df["fecha"], df["ventas"], marker="o")
plt.title("Ventas históricas - Producto")
plt.xlabel("Fecha")
plt.ylabel("Unidades vendidas")
plt.grid(True)
plt.tight_layout()
plt.show()
