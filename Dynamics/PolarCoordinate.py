"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:11 PM 
els.obrq@gmail.com

"""

import numpy as np


class PolarCoordinate(object):
    def __init__(self, dt, Isp, g_planet, mu_planet, r_planet, mass):
        self.mass = mass
        self.dt = dt
        self.Isp = Isp
        self.ge = 9.807
        self.mu = mu_planet
        self.r_moon = r_planet
        self.g_planet = g_planet
        self.c_char = Isp * self.ge
        return

    def dynamics_polar(self, state, T, psi=0):
        r       = state[0]
        v       = state[1]
        theta   = state[2]
        omega   = state[3]
        m       = state[4]

        rhs    = np.zeros(5)
        rhs[0] = v
        rhs[1] = T/m * np.sin(psi) - self.mu/(r ** 2) + r * omega ** 2
        rhs[2] = omega
        rhs[3] = -(T/m * np.cos(psi) + 2 * v * omega)/r
        rhs[4] = - T/self.c_char
        return rhs

    def rungeonestep(self, state, T, psi=0):
        x = np.array(state)
        k1 = self.dynamics_polar(x, T, psi)
        xk2 = x + (self.dt / 2.0) * k1
        k2 = self.dynamics_polar(xk2, T, psi)
        xk3 = x + (self.dt / 2.0) * k2
        k3 = self.dynamics_polar(xk3, T, psi)
        xk4 = x + self.dt * k3
        k4 = self.dynamics_polar(xk4, T, psi)
        next_x = x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_x
