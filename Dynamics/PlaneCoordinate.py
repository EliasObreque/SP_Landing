"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:11 PM 
els.obrq@gmail.com

"""

import numpy as np
ge = 9.807


class PlaneCoordinate(object):
    def __init__(self, dt, mu_planet, r_planet, mass, inertia):
        self.mass_0 = mass
        self.current_mass = mass
        self.dt = dt
        self.mu = mu_planet
        self.r_moon = r_planet
        self.delta_alpha_k = np.random.normal(0, np.deg2rad(2))
        self.delta_d_k = np.random.normal(0, np.deg2rad(2))
        self.inertia_0 = inertia # 1 / 12 * self.mass_0 * (0.2 ** 2 + 0.3 ** 2)
        self.current_inertia = self.inertia_0
        self.current_pos_i = np.zeros(2)
        self.current_vel_i = np.zeros(2)
        self.current_theta = 0
        self.current_omega = 0
        self.current_time = 0.0
        self.m_dot_p = 0.0
        self.historical_pos_i = []
        self.historical_vel_i = []
        self.historical_mass = []
        self.historical_theta = []
        self.historical_omega = []
        self.historical_inertia = []
        self.historical_time = []
        return

    def dynamic(self, state, F, tau_b, psi=0):
        r = state[0:2]
        v = state[2:4]
        m = state[4]
        theta = state[5]
        omega = state[6]
        inertia = state[7]

        u_f_i = np.array([-np.sin(theta + self.delta_alpha_k), np.cos(theta + self.delta_alpha_k)])
        rhs = np.zeros(8)
        rhs[0:2] = v
        rhs[2:4] = F / m * u_f_i - self.mu * r / (r ** 3)
        rhs[4] = - self.m_dot_p
        rhs[2] = omega
        rhs[3] = tau_b / inertia
        rhs[4] = self.inertia_0 * rhs[4] / self.mass_0
        return rhs

    def update(self, thrust_i, m_dot_p, torque_b):
        self.rungeonestep(thrust_i, m_dot_p, torque_b)

    def rungeonestep(self, T, m_dot_p, torque_b):
        x = np.concatenate([self.current_pos_i, self.current_vel_i])

        self.m_dot_p = m_dot_p
        k1 = self.dynamic(x, T, torque_b)
        xk2 = x + (self.dt / 2.0) * k1
        k2 = self.dynamic(xk2, T, torque_b)
        xk3 = x + (self.dt / 2.0) * k2
        k3 = self.dynamic(xk3, T, torque_b)
        xk4 = x + self.dt * k3
        k4 = self.dynamic(xk4, T, torque_b)
        next_x = x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.current_time += self.dt
        return next_x

    def save_data(self):
        self.historical_pos_i.append(self.current_pos_i)
        self.historical_vel_i.append(self.current_pos_i)
        self.historical_mass.append(self.current_mass)
        self.historical_theta.append(self.current_theta)
        self.historical_omega.append(self.current_omega)
        self.historical_inertia.append(self.current_inertia)
        self.historical_time.append(self.current_time)


if __name__ == '__main__':
    import numpy
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 14

    mu = 4.9048695e12  # m3s-2
    rm = 1.738e6
    ra = 68e6 + rm
    rp = 2e6 + rm
    a = 0.5 * (ra + rp)

    v0 = np.sqrt(2 * mu / ra - mu / a)
    a1 = 0.5 * (rm + ra)
    v1 = np.sqrt(2 * mu / ra - mu / a1)

