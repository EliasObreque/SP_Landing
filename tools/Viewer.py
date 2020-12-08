"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
from matplotlib import pyplot as plt


def create_plot():
    plt.figure(1)
    plt.ylabel('Radius [m]')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(2)
    plt.ylabel('Radial Velocity [m/s]')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(3)
    plt.ylabel(r'$\theta$ [rad]')
    plt.xlabel('Time [s]')
    plt.grid()

    plt.figure(4)
    plt.ylabel('Angular velocity [rad/s]')
    plt.xlabel('Time [s]]')
    plt.grid()

    plt.figure(5)
    plt.ylabel('Mass [kg]')
    plt.xlabel('Time [s]]')
    plt.grid()

    plt.figure(6)
    plt.ylabel('Thrust (N)')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.figure(7)
    plt.ylabel('Altitude [m]')
    plt.xlabel('Time [s]')
    plt.grid()

    fig = plt.figure(8)
    ax = fig.gca()
    r_moon = 1738e3
    circle1 = plt.Circle((0, 0), r_moon, color='darkgray')
    plt.ylabel('Orbit position Y [m]')
    plt.xlabel('Orbit position X [m]')
    ax.add_artist(circle1)
    plt.grid()

    plt.figure(9)
    plt.ylabel(r'$\phi$ [rad]')
    plt.xlabel('Time [s]')
    plt.grid()


def set_plot(n_figure, x, y, opt1, opt2, max_H=0, max_V=0):
    plt.figure(n_figure)
    plt.plot(x, y, opt1, lw=1)
    plt.plot(x[0], y[0], opt2, markersize=4)
