"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:11 PM 
els.obrq@gmail.com

"""

import numpy as np
from tools.mathtools import runge_kutta_4, rkf45, omega4kinematics, skewsymmetricmatrix, matrix_from_vector

ge = 9.807


class SixDoF(object):
    def __init__(self, dt, mu_planet, r_planet, mass, inertia, state):
        self.mass_0 = mass
        self.state_0 = state
        self.dt = dt
        self.mu = mu_planet
        self.r_moon = r_planet
        self.delta_alpha_k = 0#np.random.normal(0, np.deg2rad(2))
        self.delta_d_k = 0#np.random.normal(0, np.deg2rad(2))
        self.inertia_0 = inertia # 1 / 12 * self.mass_0 * (0.2 ** 2 + 0.3 ** 2)
        self.current_pos_i = state[0]
        self.current_vel_i = state[1]
        self.current_q_i2b = state[2]
        self.current_omega = state[3]
        self.current_mass = mass
        self.current_inertia = self.inertia_0
        self.current_time = 0.0
        self.m_dot_p = 0.0
        self.historical_pos_i = []
        self.historical_vel_i = []
        self.historical_mass = []
        self.historical_qi2b = []
        self.historical_omega = []
        self.historical_inertia = []
        self.historical_time = []
        self.save_data()

        self.h_old = self.dt
        return

    def reset(self):
        self.current_mass = self.mass_0
        self.current_inertia = self.inertia_0
        self.current_pos_i = self.state_0[0]
        self.current_vel_i = self.state_0[1]
        self.current_q_i2b = self.state_0[2]
        self.current_omega = self.state_0[3]
        self.current_time = 0.0
        self.m_dot_p = 0.0
        self.historical_pos_i = []
        self.historical_vel_i = []
        self.historical_mass = []
        self.historical_qi2b = []
        self.historical_omega = []
        self.historical_inertia = []
        self.historical_time = []
        self.save_data()

    def dynamic(self, state, *args):
        r = state[0:3]
        v = state[3:6]
        m = state[6]
        q_i2b = state[7:11]
        omega = state[11:14]
        inertia = matrix_from_vector(state[14:])
        inv_inertia = np.linalg.inv(inertia)
        r3 = np.linalg.norm(r) ** 3

        thrust = args[0]
        torque = args[1]

        rdot = v
        vdot = -self.mu * r / r3 + thrust / m
        mdot = args[2]
        dq = 0.5 * omega4kinematics(omega) @ q_i2b
        sk = skewsymmetricmatrix(omega)
        h_total_b = inertia.dot(omega)
        dw = - inv_inertia @ (sk @ h_total_b - torque)
        dinertia = np.zeros(6)
        return np.array([*rdot, *vdot, mdot, *dq, *dw, *dinertia])

    def update(self, thrust_i, m_dot_p, torque_b, low_step):
        x_state = np.concatenate([self.current_pos_i,
                                  self.current_vel_i,
                                  self.current_mass,
                                  self.current_q_i2b,
                                  self.current_omega,
                                  self.current_inertia])
        if low_step is not None:
            new_var = x_state + runge_kutta_4(self.dynamic, x_state, low_step, (thrust_i, torque_b, m_dot_p))
            h = low_step
        else:
            new_var, h = rkf45(self.dynamic, x_state, low_step, self.current_time, np.inf, (thrust_i, torque_b, m_dot_p))
        self.current_pos_i = new_var[:2]
        self.current_vel_i = new_var[2:4]
        self.current_mass = new_var[4]
        self.current_q_i2b = new_var[5]
        self.current_omega = new_var[6]
        self.current_time += h

    def save_data(self):
        self.historical_pos_i.append(self.current_pos_i)
        self.historical_vel_i.append(self.current_vel_i)
        self.historical_mass.append(self.current_mass)
        self.historical_qi2b.append(self.current_q_i2b)
        self.historical_omega.append(self.current_omega)
        self.historical_inertia.append(self.current_inertia)
        self.historical_time.append(self.current_time)

    def get_state_idx(self, idx):
        return [self.historical_pos_i[idx],
                self.historical_vel_i[idx],
                self.historical_mass[idx],
                self.historical_qi2b[idx],
                self.historical_omega[idx],
                self.historical_inertia[idx],
                self.historical_time[idx]]

    def get_historial(self):
        return [self.historical_pos_i,
                self.historical_vel_i,
                self.historical_mass,
                self.historical_qi2b,
                self.historical_omega,
                self.historical_inertia,
                self.historical_time]

    def get_current_state(self):
        return [self.current_pos_i, self.current_vel_i,
                self.current_q_i2b, self.current_omega,
                self.current_mass, self.current_inertia]


if __name__ == '__main__':
    pass

