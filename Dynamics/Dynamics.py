"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""

import numpy as np


class Dynamics(object):
    def __init__(self, dt, Isp, g_planet):
        self.step_width = dt
        self.current_time = 0
        self.Isp = Isp
        self.ge = 9.807
        self.g_planet = g_planet
        self.c_char = Isp * self.ge
        self.alpha = 1

    def update_control_parameters(self, m0, current_thrust):
        self.alpha = current_thrust/self.c_char
        self.a = 0.5 * (self.c_char * self.alpha - np.linalg.norm(self.g_planet) * m0) / m0
        self.b = 0.5 * self.c_char * self.alpha ** 2 / (m0 ** 2)

    def control_sf(self, alt, vel):
        sf = (self.b / self.a) * alt + 2 * self.a * np.sqrt(alt / self.a) + vel
        return np.abs(sf)/sf

    def dynamics(self, state, t, T):
        x = state[0]
        vx = state[1]
        mass = state[2]
        rhs = np.zeros(3)
        rhs[0] = vx
        rhs[1] = self.g_planet + T / mass
        rhs[2] = -T / self.c_char
        return rhs

    def rungeonestep(self, T, pos, vel, mass):
        t = self.current_time
        dt = self.step_width
        x = np.array([pos, vel, mass])
        k1 = self.dynamics(x, t, T)
        xk2 = x + (dt / 2.0) * k1
        k2 = self.dynamics(xk2, (t + dt / 2.0), T)
        xk3 = x + (dt / 2.0) * k2
        k3 = self.dynamics(xk3, (t + dt / 2.0), T)
        xk4 = x + dt * k3
        k4 = self.dynamics(xk4, (t + dt), T)
        next_x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.current_time += self.step_width
        return next_x