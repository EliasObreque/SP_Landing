"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:11 PM 
els.obrq@gmail.com

"""

import numpy as np
ge = 9.807
a = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
b = np.array([[0, 0, 0, 0, 0],
              [1 / 4, 0, 0, 0, 0],
              [3 / 32, 9 / 32, 0, 0, 0],
              [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
              [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
              [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]])
c4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
c5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])


class PlaneCoordinate(object):
    def __init__(self, dt, mu_planet, r_planet, mass, inertia, state):
        self.mass_0 = mass
        self.state_0 = state
        self.current_mass = mass
        self.dt = dt
        self.mu = mu_planet
        self.r_moon = r_planet
        self.delta_alpha_k = 0#np.random.normal(0, np.deg2rad(2))
        self.delta_d_k = 0#np.random.normal(0, np.deg2rad(2))
        self.inertia_0 = inertia # 1 / 12 * self.mass_0 * (0.2 ** 2 + 0.3 ** 2)
        self.current_inertia = self.inertia_0
        self.current_pos_i = state[0]
        self.current_vel_i = state[1]
        self.current_theta = state[2]
        self.current_omega = state[3]
        self.current_time = 0.0
        self.m_dot_p = 0.0
        self.historical_pos_i = []
        self.historical_vel_i = []
        self.historical_mass = []
        self.historical_theta = []
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
        self.current_theta = self.state_0[2]
        self.current_omega = self.state_0[3]
        self.current_time = 0.0
        self.m_dot_p = 0.0
        self.historical_pos_i = []
        self.historical_vel_i = []
        self.historical_mass = []
        self.historical_theta = []
        self.historical_omega = []
        self.historical_inertia = []
        self.historical_time = []
        self.save_data()

    def dynamic(self, state, F, tau_b, psi=0):
        r = state[0:2]
        v = state[2:4]
        m = state[4]
        theta = state[5]
        omega = state[6]
        inertia = state[7]
        u_f_i = np.array([-np.sin(theta + self.delta_alpha_k), np.cos(theta + self.delta_alpha_k)])
        rhs = np.zeros(8)
        u_f_i = -v / np.linalg.norm(v)
        rhs[0:2] = v
        rhs[2:4] = F / m * u_f_i - self.mu * r / (np.linalg.norm(r) ** 3)
        rhs[4] = - self.m_dot_p
        rhs[5] = omega
        rhs[6] = tau_b / inertia
        rhs[7] = self.inertia_0 * rhs[4] / self.mass_0
        return rhs

    def update(self, thrust_i, m_dot_p, torque_b, low_step):
        if low_step is not None:
            new_var = self.rungeonestep(thrust_i, m_dot_p, torque_b)
        else:
            new_var = self.rkf45(thrust_i, m_dot_p, torque_b)
        self.current_pos_i = new_var[:2]
        self.current_vel_i = new_var[2:4]
        self.current_mass = new_var[4]
        self.current_theta = new_var[5]
        self.current_omega = new_var[6]

    def rungeonestep(self, T, m_dot_p, torque_b=np.array([0, 0, 0])):
        x = np.concatenate([self.current_pos_i, self.current_vel_i, np.array([self.current_mass, self.current_theta,
                                                                              self.current_omega, self.current_inertia])])

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
        self.historical_vel_i.append(self.current_vel_i)
        self.historical_mass.append(self.current_mass)
        self.historical_theta.append(self.current_theta)
        self.historical_omega.append(self.current_omega)
        self.historical_inertia.append(self.current_inertia)
        self.historical_time.append(self.current_time)

    def get_state_idx(self, idx):
        return [self.historical_pos_i[idx],
                self.historical_vel_i[idx],
                self.historical_mass[idx],
                self.historical_theta[idx],
                self.historical_omega[idx],
                self.historical_inertia[idx],
                self.historical_time[idx]]

    def get_historial(self):
        return [self.historical_pos_i,
                self.historical_vel_i,
                self.historical_mass,
                self.historical_theta,
                self.historical_omega,
                self.historical_inertia,
                self.historical_time]

    def rkf45(self, thrust_i, m_dot_p, torque_b, tol=1e-15):
        x = np.concatenate([self.current_pos_i, self.current_vel_i, np.array([self.current_mass, self.current_theta,
                                                                              self.current_omega,
                                                                              self.current_inertia])])
        self.m_dot_p = m_dot_p
        beta = 0.9
        hmin = 0.1
        h = self.h_old
        k_i = np.zeros(len(x))
        x_out = np.zeros(len(x))
        k_nn = np.tile(k_i, (6, 1))
        end_condition = False
        while end_condition is False:
            for i in range(6):
                t_inner = self.current_time + a[i] * h
                x_inner = x
                for j in range(i):
                    x_inner = x_inner + h * k_nn[j, :] * b[i][j]
                k_nn[i, :] = self.dynamic(x_inner, thrust_i, torque_b)

            te = h * np.dot(c4 - c5, k_nn)
            # print(k_nn)
            # print(n_ite, x_k_max, x_k_min)
            error = np.max(np.abs(te))
            xmax = np.max(np.abs(x))
            te_allowed = tol * max(xmax, 1.0)
            delta = (te_allowed / (error + np.finfo(np.float64).eps)) ** (1.0 / 5.0)
            if error <= te_allowed:
                end_condition = True
                x_out = x + h * np.dot(c5, k_nn)
                self.current_time += h
            h = min(beta * delta * h, 4 * h)
            if h < hmin:
                raise print('Warning: Step size fell below its minimum allowable value {}'.format(h))
        self.h_old = h
        self.dt = h
        return x_out


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

