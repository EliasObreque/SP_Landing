"""
Created by Elias Obreque
Date: 14-08-2023
email: els.obrq@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Generar datos de ejemplo en 3D
np.random.seed(0)
n_samples = 20
X = np.random.rand(n_samples, 3)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] + 0.1 * np.random.randn(n_samples)

# Crear el modelo de Kriging con kernel RBF y regularización espacial
kernel = 1.0 * RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-1, 10.0))
alpha = 1.0  # Factor de regularización espacial
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=alpha)

# Ajustar el modelo a los datos
gp.fit(X, y)

# Generar puntos de prueba en 3D
x_pred = np.random.rand(1000, 3)

# Realizar predicciones
y_pred, sigma = gp.predict(x_pred, return_std=True)

# Graficar los resultados en 3D
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], y, c='r', label='Datos de ejemplo')
ax1.scatter(x_pred[:, 0], x_pred[:, 1], y_pred, c='b', marker='.', label='Predicción')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('y')
ax1.set_title('Interpolación y Extrapolación con Kriging en 3D')
ax1.legend()

# Graficar heatmap en los puntos de predicción
ax2 = fig.add_subplot(122)
sc = ax2.scatter(x_pred[:, 0], x_pred[:, 1], c=y_pred, cmap='viridis', marker='o', s=50)
plt.colorbar(sc, ax=ax2, label='Predicción en los puntos')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_title('Heatmap en los puntos de predicción')

plt.tight_layout()
plt.show()
