"""
Created by Elias Obreque
Date: 28-11-2023
email: els.obrq@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from io import BytesIO
import matplotlib.image as mpimg
from PIL import Image
rm = 1.738e6

# Descargar y cargar los datos de elevación lunar
data = Image.open("moon-58-1024x1024.png")
dims = data.size
factor = int(rm / dims[0] * 0.5)
data = np.asarray(data)[61:-75, 68:-68]
data_ = Image.fromarray(data)
data_ = data_.resize((int(2 * rm * 1e-3), int(2 * rm * 1e-3)))

# Crear una figura y un eje con proyección PlateCarree
fig, ax = plt.subplots()
ax.imshow(data_)

plt.show()
