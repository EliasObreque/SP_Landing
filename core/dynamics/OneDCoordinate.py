"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:11 PM 
els.obrq@gmail.com

"""
import numpy as np
ge = 9.807


class LinearCoordinate(object):
    def __init__(self, dt, g_planet, mass):
        self.dt = dt
        self.Isp = None
        self.g_planet = g_planet
        self.x_fixed = None
        self.m_dot_p = 0.0

    def dynamics_1d(self, state, T):
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
        rhs[2] = -self.m_dot_p
        return rhs

    def rungeonestep(self, current_x, thrust, current_m_dot_p, torque_b=None):
        self.m_dot_p = current_m_dot_p
        # At each step of the Runge consider constant thrust
        self.x_fixed = np.array(current_x)
        if self.x_fixed[0] <= 0 and self.x_fixed[1] < 0:
            self.x_fixed[1] = 0
        x1 = self.x_fixed
        k1 = self.dynamics_1d(x1, thrust)
        xk2 = x1 + (self.dt / 2.0) * k1
        k2 = self.dynamics_1d(xk2, thrust)
        xk3 = x1 + (self.dt / 2.0) * k2
        k3 = self.dynamics_1d(xk3, thrust)
        xk4 = x1 + self.dt * k3
        k4 = self.dynamics_1d(xk4, thrust)
        next_x = x1 + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_x

    def get_current_state(self):
        return [self.current_pos_i, self.current_vel_i, self.current_mass]
