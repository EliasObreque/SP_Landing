"""
Created by Elias Obreque
Date: 03-01-2024
email: els.obrq@gmail.com
"""

import matplotlib.pyplot as plt

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

# Sample data
x = [1, 2, 3, 4, 5]
y1 = [0, 20, 30, 20, 50]
y2 = [1, 0, 3, 2, 15]

# Create subplots
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plot data on ax1 and ax2
ax1.plot(x, y1, color='blue', label='y1')
ax1.grid()
ax2.plot(x, y2, color='red', label='y2')
ax2.grid()
# Align y-axes
align_yaxis(ax1, 30, ax2, 3)  # Aligning value 30 on ax1 with value 2 on ax2

# Show legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plots
plt.show()

