"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""

import numpy as np
from matplotlib import pyplot as plt
from tools.GeneticAlgorithm import GeneticAlgorithm


class Dynamics(object):
    def __init__(self, dt, Isp, g_planet, mass, alpha_min, alpha_max):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_med = (alpha_max + alpha_min) * 0.5
        self.mass = mass
        self.step_width = dt
        self.current_time = 0
        self.Isp = Isp
        self.ge = 9.807
        self.g_planet = g_planet
        self.c_char = Isp * self.ge
        self.alpha = 1
        self.T_max = alpha_max * self.c_char
        self.T_min = alpha_min * self.c_char
        self.x2_hat_min = 0
        self.x2_hat_max = 0
        self.x2_hat_med = 0
        self.tolerance = 1e-6

    def control_sf(self, alt, vel, mass, thrust):
        current_alpha = thrust / self.c_char
        flag_1 = self.calc_first_cond(alt, vel, current_alpha)
        sf = vel - self.calc_x2_char(current_alpha, self.t1)
        return np.abs(sf) / sf

    def calc_first_cond(self, alt, vel, alpha):
        return np.abs(vel) - alt * alpha / self.mass

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

    def calc_x2_char(self, alpha, t1):
        g = -self.g_planet
        arg_ = 1 - alpha * t1 / self.mass
        return self.c_char * np.log(arg_) + g * t1

    def calc_x1_char(self, alpha, t1, x2_c=None):
        g = -self.g_planet
        arg_ = 1 - alpha * t1 / self.mass
        if x2_c is None:
            x2_c = self.calc_x2_char(alpha, t1)
        return 0.5 * g * t1 ** 2 - self.c_char * t1 - self.c_char * self.mass * arg_ * np.log(arg_) / alpha  - x2_c * t1

    def calc_t1(self, x1, x2, x3, alpha):
        A = -0.5*self.g_planet * alpha
        B = (self.c_char * alpha + self.g_planet * x3)
        C = x1 * alpha + x2 * x3
        t1 = (-B + np.sqrt(B**2 - 4*A*C))/(2 * A)
        return t1

    def calc_limits(self, t1):
        self.t1 = t1
        g = -self.g_planet
        # Minimum
        self.x2_char_min = self.calc_x2_char(self.alpha_min, self.t1)
        self.x1_char_min = self.calc_x1_char(self.alpha_min, self.t1, self.x2_char_min)
        # Maximum
        self.x2_char_max = self.calc_x2_char(self.alpha_max, self.t1)
        self.x1_char_max = self.calc_x1_char(self.alpha_max, self.t1, self.x2_char_max)
        # Medium
        self.x2_char_med = self.calc_x2_char(self.alpha_med, self.t1)
        self.x1_char_med = self.calc_x1_char(self.alpha_med, self.t1, self.x2_char_med)

        print("Minimum state parameters *")
        print(self.x2_char_min, " [m/s] - ", self.x1_char_min, " [m]\n")
        print("Medium state parameters * ")
        print(self.x2_char_med, " [m/s] - ", self.x1_char_med, " [m]\n")
        print("Maximum state parameters * ")
        print(self.x2_char_max, " [m/s] - ",  self.x1_char_max, " [m]\n")

        self.t2_min = - self.x2_char_min / g
        self.t2_max = - self.x2_char_max / g
        self.t2_med = - self.x2_char_med / g
        self.x1_hat_min = self.x1_char_min + 0.5 * self.x2_char_min ** 2 / g
        self.x1_hat_max = self.x1_char_max + 0.5 * self.x2_char_max ** 2 / g
        self.x1_hat_med = self.x1_char_med + 0.5 * self.x2_char_med ** 2 / g
        print("Initial state [min, max] [m]")
        print("[", self.x1_hat_min, " ,", self.x1_hat_max, "]")
        return

    def show_limits(self):
        g = -self.g_planet
        t1 = np.linspace(0, self.t1, 200)
        t2_min = np.linspace(0, self.t2_min)
        t2_max = np.linspace(0, self.t2_max)
        t2_med = np.linspace(0, self.t2_med)

        # Minimum
        x2_min = self.calc_x2_char(self.alpha_min, t1)
        x1_min = self.calc_x1_char(self.alpha_min, t1, x2_min)
        # Maximum
        x2_max = self.calc_x2_char(self.alpha_max, t1)
        x1_max = self.calc_x1_char(self.alpha_max, t1, x2_max)
        # Medium
        x2_med = self.calc_x2_char(self.alpha_med, t1)
        x1_med = self.calc_x1_char(self.alpha_med, t1, x2_med)

        x2_hat_min = self.x2_hat_min - t2_min * g
        x2_hat_max = self.x2_hat_max - t2_max * g
        x2_hat_med = self.x2_hat_med - t2_med * g

        x1_hat_min = self.x1_hat_min - 0.5 * x2_hat_min ** 2 / g
        x1_hat_max = self.x1_hat_max - 0.5 * x2_hat_max ** 2 / g
        x1_hat_med = self.x1_hat_med - 0.5 * x2_hat_med ** 2 / g

        plt.figure()
        plt.grid()
        plt.xlabel("Altitude [m]")
        plt.ylabel("Velocity [m/s]")
        plt.plot(x1_hat_min, x2_hat_min, '--k', label='free-fall')
        plt.plot(x1_min, x2_min, 'k', label='sf')
        plt.plot(x1_hat_max, x2_hat_max, '--r', label='free-fall')
        plt.plot(x1_max, x2_max, 'r', label='sf')
        plt.plot(x1_hat_med, x2_hat_med, '--b', label='free-fall')
        plt.plot(x1_med, x2_med, 'b', label='sf')
        plt.legend()
        plt.show()
        return

    def calc_optimal_parameters(self, max_generation, n_variables, n_individuals, range_variables):
        ga = GeneticAlgorithm(max_generation, n_variables, n_individuals, range_variables)
        ga.optimize()
        return

    def calc_simple_optimal_parameters(self, r0):
        self.tar_r0 = r0
        return self.calc_variable()

    def bisection_method(self, var_left, var_right):
        f_l = self.tar_r0 - self.calc_x1_char(var_left, self.t1) + 0.5 * self.calc_x2_char(var_left, self.t1) ** 2 / self.g_planet
        f_r = self.tar_r0 - self.calc_x1_char(var_right, self.t1) + 0.5 * self.calc_x2_char(var_right, self.t1) ** 2 / self.g_planet
        return -f_l, -f_r

    def calc_variable(self):
        a_cur = self.alpha_min
        b_cur = self.alpha_max
        f_l, f_r = self.bisection_method(a_cur, b_cur)
        c_cur = (a_cur + b_cur)/2
        f_c, _ = self.bisection_method(c_cur, 5)
        c_last = c_cur
        while np.abs(f_c) > self.tolerance:
            if f_c < 0 and f_l < 0:
                a_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur)
            elif f_c < 0 < f_l:
                b_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur)
            elif f_c > 0 and f_r > 0:
                b_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur)
            else:
                a_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur)
            c_cur = (a_cur + b_cur)/2
            f_c, _ = self.bisection_method(c_cur, 5)
            c_last = c_cur
        return c_cur
