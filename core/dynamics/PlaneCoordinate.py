"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:11 PM 
els.obrq@gmail.com

"""

import numpy as np
from tools.mathtools import runge_kutta_4, rkf45
ge = 9.807


class PlaneCoordinate(object):
    def __init__(self, dt, mu_planet, r_planet, mass, inertia, state):
        self.mass_0 = mass
        self.state_0 = state
        self.current_mass = mass
        self.dt = dt
        self.mu = mu_planet
        self.r_moon = r_planet
        # self.delta_alpha_k = np.random.normal(0, np.deg2rad(1))
        # self.delta_d_k = np.random.normal(0, np.deg2rad(1))
        self.inertia_0 = inertia
        self.current_inertia = self.inertia_0
        self.current_pos_i = state[0]
        self.current_vel_i = state[1]
        self.current_theta = state[2]
        self.current_omega = state[3]
        self.current_time = 0.0
        self.current_thrust = 0.0
        self.current_torque = 0.0
        self.current_energy = self.get_energy()
        self.m_dot_p = 0.0
        self.historical_pos_i = []
        self.historical_vel_i = []
        self.historical_mass = []
        self.historical_theta = []
        self.historical_omega = []
        self.historical_inertia = []
        self.historical_time = []
        self.historical_thrust = []
        self.historical_energy = []
        self.historical_torque = []
        self.save_data()
        self.h_old = self.dt

    def get_energy(self):
        return 0.5 * np.linalg.norm(self.current_vel_i) ** 2 - self.mu / np.linalg.norm(self.current_pos_i)

    def reset(self):
        self.current_mass = self.mass_0
        self.current_inertia = self.inertia_0
        self.current_pos_i = self.state_0[0]
        self.current_vel_i = self.state_0[1]
        self.current_theta = self.state_0[2]
        self.current_omega = self.state_0[3]
        self.current_time = 0.0
        self.current_torque = 0.0
        self.current_thrust = 0.0
        self.m_dot_p = 0.0
        self.historical_pos_i = []
        self.historical_vel_i = []
        self.historical_mass = []
        self.historical_theta = []
        self.historical_omega = []
        self.historical_inertia = []
        self.historical_time = []
        self.historical_thrust = []
        self.historical_energy = []
        self.historical_torque = []
        self.save_data()

    def dynamic(self, state, ct, *args):
        r = state[0:2]
        v = state[2:4]
        m = state[4]
        theta = state[5]
        omega = state[6]
        inertia = state[7]

        thrust = args[0]
        torque = args[1]

        u_f_i = np.array([-np.sin(theta), np.cos(theta)])
        rhs = np.zeros(8)
        # u_f_i = -v / np.linalg.norm(v)
        rhs[0:2] = v
        rhs[2:4] = thrust / m * u_f_i - self.mu * r / (np.linalg.norm(r) ** 3)
        rhs[4] = - args[2] if thrust > 0 else 0
        rhs[5] = omega
        rhs[6] = torque / inertia
        rhs[7] = self.inertia_0 * rhs[4] / self.mass_0
        return rhs

    def update(self, thrust_i, m_dot_p, torque_b, low_step):
        x_state = np.stack([self.current_pos_i[0],
                            self.current_pos_i[1],
                            self.current_vel_i[0],
                            self.current_vel_i[1],
                            self.current_mass,
                            self.current_theta,
                            self.current_omega,
                            self.current_inertia])
        self.current_thrust = thrust_i
        self.current_torque = torque_b
        if low_step is not None:
            new_var = x_state + runge_kutta_4(self.dynamic, x_state, low_step, self.current_time, thrust_i, torque_b, m_dot_p)
            self.dt = low_step
            h = low_step
        else:
            new_var, h = rkf45(self.dynamic, x_state, self.dt, self.current_time, np.inf,
                               thrust_i, torque_b, m_dot_p)
            self.dt = h
        self.current_pos_i = new_var[:2]
        self.current_vel_i = new_var[2:4]
        self.current_mass = new_var[4]
        self.current_theta = new_var[5] % (2 * np.pi)
        self.current_omega = new_var[6]
        self.current_inertia = new_var[7]
        self.current_time += h

    def save_data(self):
        self.historical_pos_i.append(self.current_pos_i)
        self.historical_vel_i.append(self.current_vel_i)
        self.historical_mass.append(self.current_mass)
        self.historical_theta.append(self.current_theta)
        self.historical_omega.append(self.current_omega)
        self.historical_inertia.append(self.current_inertia)
        self.historical_time.append(self.current_time)
        self.historical_thrust.append(self.current_thrust)
        self.historical_torque.append(self.current_torque)
        self.historical_energy.append(self.get_energy())

    def get_state_idx(self, idx):
        return [self.historical_pos_i[idx],
                self.historical_vel_i[idx],
                self.historical_mass[idx],
                self.historical_theta[idx],
                self.historical_omega[idx],
                self.historical_inertia[idx],
                self.historical_time[idx],
                self.historical_thrust[idx],
                self.historical_torque[idx],
                self.historical_energy[idx]]

    def get_historial(self):
        return [self.historical_pos_i,
                self.historical_vel_i,
                self.historical_mass,
                self.historical_theta,
                self.historical_omega,
                self.historical_inertia,
                self.historical_thrust,
                self.historical_torque,
                self.historical_energy,
                self.historical_time]

    def get_current_state(self):
        return [self.current_pos_i, self.current_vel_i, self.current_theta, self.current_omega,
                self.current_mass, self.current_inertia]


if __name__ == '__main__':
    pass

