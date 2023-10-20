"""
Created by Elias Obreque
Date: 14-08-2023
email: els.obrq@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
rm = 1.738e6
rp = 2e6
mu = 4.9048695e12  # m3s-2


def get_energy(mu, r, v):
    return 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)


# Definir la resolución de la superficie
res = 100
xi = np.linspace(rm, rp, res)
yi = np.linspace(0, 2000, res)
xi, yi = np.meshgrid(xi, yi)

z = 0.5 * yi ** 2 - mu / xi
z_min = np.min(z)
z_max = np.max(z)

fig, ax = plt.subplots()

c = ax.pcolormesh(xi, yi, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([xi.min(), xi.max(), yi.min(), yi.max()])
fig.colorbar(c, ax=ax)

plt.show()


