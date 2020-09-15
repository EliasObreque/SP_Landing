"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
from matplotlib import pyplot as plt


def create_plot():
    plt.figure(1)
    plt.ylabel('Position (m)')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(2)
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(3)
    plt.xlabel('x_1 [m]')
    plt.ylabel('x_2 [m/s]')
    plt.grid()

    plt.figure(4)
    plt.ylabel('Mass (kg)')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(5)
    plt.ylabel('Thrust (N)')
    plt.xlabel('Time (s)')
    plt.grid()


def set_plot(n_figure, x, y, opt1, opt2):
    plt.figure(n_figure)
    plt.plot(x, y, opt1, markersize=4)
    plt.plot(x[0], y[0], opt2, markersize=8)