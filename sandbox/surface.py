"""
Created by Elias Obreque
Date: 14-08-2023
email: els.obrq@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Tus puntos en coordenadas x, y, z
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
z = np.array([5, 6, 7, 8, 9])

# Definir la resolución de la superficie
res = 50
xi = np.linspace(min(x), max(x), res)
yi = np.linspace(min(y), max(y), res)
xi, yi = np.meshgrid(xi, yi)

# Interpolación de los puntos para obtener valores z en la superficie
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(xi, yi, zi, cmap='viridis')

# Agregar puntos originales
ax.scatter(x, y, z, color='red', s=50, label='Puntos Originales')

# Personalizar la apariencia
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Superficie Aproximada')
ax.legend()

# Mostrar el gráfico
plt.show()

