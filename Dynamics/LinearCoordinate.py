"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:11 PM 
els.obrq@gmail.com

"""
import numpy as np
ge = 9.807


class LinearCoordinate(object):
    def __init__(self, dt, Isp, g_planet, mass):
        self.dt = dt
        self.Isp = Isp
        self.g_planet = g_planet
        self.c_char = Isp * ge
        self.x_fixed = None
        return

    def dynamics_1d(self, state, T, psi=0):
        alt = state[0]
        vx = state[1]
        mass = state[2]
        rhs = np.zeros(3)
        rhs[0] = vx
        if self.x_fixed[0] > 0.0:
            rhs[1] = self.g_planet + T / mass
        else:
            if alt > self.x_fixed[0]:
                rhs[1] = self.g_planet + T / mass
            else:
                rhs[0] = 0 if vx < 0 else vx
                rhs[1] = 0
                if np.abs(self.g_planet) < np.abs(T / mass):
                    rhs[1] = self.g_planet + T / mass
        rhs[2] = -T / self.c_char
        return rhs

    def rungeonestep(self, state, thrust, psi=0):
        # At each step of the Runge consider constant thrust
        self.x_fixed = np.array(state)
        if self.x_fixed[0] <= 0 and self.x_fixed[1] < 0:
            self.x_fixed[1] = 0
        x1 = self.x_fixed
        k1 = self.dynamics_1d(x1, thrust, psi)
        xk2 = x1 + (self.dt / 2.0) * k1
        k2 = self.dynamics_1d(xk2, thrust, psi)
        xk3 = x1 + (self.dt / 2.0) * k2
        k3 = self.dynamics_1d(xk3, thrust, psi)
        xk4 = x1 + self.dt * k3
        k4 = self.dynamics_1d(xk4, thrust, psi)
        next_x = x1 + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_x
