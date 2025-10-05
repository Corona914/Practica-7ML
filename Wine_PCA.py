# Wine_PCA.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set(style='whitegrid')

#  dataset "Wine Quality" (ID 186)
wine_quality = fetch_ucirepo(id=186)

# Extraemos las características (X) y el objetivo (y)
X = wine_quality.data.features
y = wine_quality.data.targets['quality'] # Es una columna llamada 'quality'

# Guardamos y MOSTRAMOS los nombres de las características para evitar errores
feature_names = X.columns.tolist()
print("--- Características Disponibles en el Dataset ---")
print(feature_names)
print("-" * 42)


#cantidad de datos original
print("--- Información del Dataset Wine Quality ---")
print(f"Dimensiones de los datos originales: {X.shape}")
print(f"Número total de muestras (vinos): {X.shape[0]}")
print(f"Número total de características por vino: {X.shape[1]}")
print("-" * 42)

# Contamos cuántas muestras hay por cada puntuación de calidad
print("Distribución de la calidad del vino:")
print(y.value_counts().sort_index())
print("-" * 42)


# Escalado de Datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación del Algoritmo PCA
print("Aplicando PCA...")
n_components = 2 # Vamos a reducir a 2 dimensiones para visualizar
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled) # Aplicamos PCA a los datos escalados
print(f"Dimensiones después de PCA: {X_pca.shape}")

#GRAFICOS
#DATOS ORIGEN
feature_1_name = 'volatile_acidity'
feature_2_name = 'alcohol'

X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_scaled_df[feature_1_name],
    y=X_scaled_df[feature_2_name],
    hue=y, # Coloreamos por la puntuación de calidad
    palette='inferno',
    alpha=0.7
)
plt.title('Dataset Original (2 de 11 Características)', fontsize=16)
plt.xlabel(f'{feature_1_name} (Estandarizado)')
plt.ylabel(f'{feature_2_name} (Estandarizado)')
plt.legend(title='Calidad del Vino')
plt.grid(True)
plt.show()

# GRÁFICO RESULTADO DE PCA
plt.figure(figsize=(10, 8))

sns.scatterplot(
    x=X_pca[:, 0], # Eje X es el primer componente principal
    y=X_pca[:, 1], # Eje Y es el segundo componente principal
    hue=y, 
    alpha=0.7
)

plt.title('Resultado de PCA en el Dataset Wine Quality', fontsize=16)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Calidad del Vino')
plt.grid(True)
plt.show()