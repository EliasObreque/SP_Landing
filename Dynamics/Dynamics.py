"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""

import numpy as np
from matplotlib import pyplot as plt
from tools.GeneticAlgorithm import GeneticAlgorithm


class Dynamics(object):
    def __init__(self, dt, Isp, g_planet, mass, alpha_min, alpha_max, t1_min, t1_max, polar_system):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.t1_min = t1_min
        self.t1_max = t1_max
        self.alpha_med = (alpha_max + alpha_min) * 0.5
        self.t1_med = (t1_max + t1_min) * 0.5
        self.mass = mass
        self.step_width = dt
        self.current_time = 0
        self.Isp = Isp
        self.ge = 9.807
        self.mu = 4.9048695e12
        self.r_moon = 1738e3
        self.g_planet = g_planet
        self.c_char = Isp * self.ge
        self.alpha = 1
        self.polar_system = polar_system
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
        sf += alt - self.calc_x1_char(current_alpha, self.t1)
        return np.abs(sf) / sf

    def calc_first_cond(self, alt, vel, alpha):
        return np.abs(vel) - alt * alpha / self.mass

    def dynamics1D(self, state, t, T, psi=0):
        x = state[0]
        vx = state[1]
        mass = state[2]
        rhs = np.zeros(3)
        rhs[0] = vx
        if x > 0.0:
            rhs[1] = self.g_planet + T / mass
        else:
            if self.g_planet < - T / mass:
                rhs[1] = 0
            else:
                rhs[1] = self.g_planet + T / mass
        rhs[2] = -T / self.c_char
        return rhs

    def dynamics_polar(self, state, t, T, psi=0):
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

    def rungeonestep(self, T, state, psi=0):
        t = self.current_time
        dt = self.step_width
        if self.polar_system:
            dynamics_selected = self.dynamics_polar
        else:
            dynamics_selected = self.dynamics1D
        x = np.array(state)
        k1 = dynamics_selected(x, t, T, psi)
        xk2 = x + (dt / 2.0) * k1
        k2 = dynamics_selected(xk2, (t + dt / 2.0), T, psi)
        xk3 = x + (dt / 2.0) * k2
        k3 = dynamics_selected(xk3, (t + dt / 2.0), T, psi)
        xk4 = x + dt * k3
        k4 = dynamics_selected(xk4, (t + dt), T, psi)
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

    def calc_limits_const_time(self, t1):
        self.t1 = t1
        # g = -self.g_planet
        # # Minimum
        # self.x2_char_min = self.calc_x2_char(self.alpha_min, self.t1)
        # self.x1_char_min = self.calc_x1_char(self.alpha_min, self.t1, self.x2_char_min)
        # # Maximum
        # self.x2_char_max = self.calc_x2_char(self.alpha_max, self.t1)
        # self.x1_char_max = self.calc_x1_char(self.alpha_max, self.t1, self.x2_char_max)
        # # Medium
        # self.x2_char_med = self.calc_x2_char(self.alpha_med, self.t1)
        # self.x1_char_med = self.calc_x1_char(self.alpha_med, self.t1, self.x2_char_med)
        #
        # print("Minimum state parameters *")
        # print(self.x2_char_min, " [m/s] - ", self.x1_char_min, " [m]\n")
        # print("Medium state parameters * ")
        # print(self.x2_char_med, " [m/s] - ", self.x1_char_med, " [m]\n")
        # print("Maximum state parameters * ")
        # print(self.x2_char_max, " [m/s] - ",  self.x1_char_max, " [m]\n")
        #
        # self.t2_min = - self.x2_char_min / g
        # self.t2_max = - self.x2_char_max / g
        # self.t2_med = - self.x2_char_med / g
        # self.x1_hat_min = self.x1_char_min + 0.5 * self.x2_char_min ** 2 / g
        # self.x1_hat_max = self.x1_char_max + 0.5 * self.x2_char_max ** 2 / g
        # self.x1_hat_med = self.x1_char_med + 0.5 * self.x2_char_med ** 2 / g
        # print("Initial state [min, max] [m]")
        # print("[", self.x1_hat_min, " ,", self.x1_hat_max, "]")
        return

    def calc_limits_const_alpha(self, alpha):
        self.alpha_const = alpha
        g = -self.g_planet
        # Minimum
        self.x2_char_min_a = self.calc_x2_char(alpha, self.t1_min)
        self.x1_char_min_a = self.calc_x1_char(alpha, self.t1_min, self.x2_char_min_a)
        # Maximum
        self.x2_char_max_a = self.calc_x2_char(alpha, self.t1_max)
        self.x1_char_max_a = self.calc_x1_char(alpha, self.t1_max, self.x2_char_max_a)
        # Medium
        self.x2_char_med_a = self.calc_x2_char(alpha, self.t1_med)
        self.x1_char_med_a = self.calc_x1_char(alpha, self.t1_med, self.x2_char_med_a)

        print("Minimum state parameters *")
        print(self.x2_char_min_a, " [m/s] - ", self.x1_char_min_a, " [m]\n")
        print("Medium state parameters * ")
        print(self.x2_char_med_a, " [m/s] - ", self.x1_char_med_a, " [m]\n")
        print("Maximum state parameters * ")
        print(self.x2_char_max_a, " [m/s] - ",  self.x1_char_max_a, " [m]\n")

        self.t2_min_a = - self.x2_char_min_a / g
        self.t2_max_a = - self.x2_char_max_a / g
        self.t2_med_a = - self.x2_char_med_a / g
        self.x1_hat_min_a = self.x1_char_min_a + 0.5 * self.x2_char_min_a ** 2 / g
        self.x1_hat_max_a = self.x1_char_max_a + 0.5 * self.x2_char_max_a ** 2 / g
        self.x1_hat_med_a = self.x1_char_med_a + 0.5 * self.x2_char_med_a ** 2 / g
        print("Initial state [min, max] [m]")
        print("[", self.x1_hat_min_a, " ,", self.x1_hat_max_a, "]")
        return

    def show_limits(self):
        g = -self.g_planet
        t1 = np.linspace(0, self.t1, 200)
        t1_max = np.linspace(0, self.t1_max, 200)
        t1_med = np.linspace(0, self.t1_med, 200)
        t1_min = np.linspace(0, self.t1_min, 200)

        t2_min = np.linspace(0, self.t2_min)
        t2_max = np.linspace(0, self.t2_max)
        t2_med = np.linspace(0, self.t2_med)
        t2_min_a = np.linspace(0, self.t2_min_a)
        t2_max_a = np.linspace(0, self.t2_max_a)
        t2_med_a = np.linspace(0, self.t2_med_a)

        # Minimum
        x2_min = self.calc_x2_char(self.alpha_min, t1)
        x1_min = self.calc_x1_char(self.alpha_min, t1, x2_min)
        x2_min_a = self.calc_x2_char(self.alpha_const, t1_min)
        x1_min_a = self.calc_x1_char(self.alpha_const, t1_min, x2_min_a)

        # Maximum
        x2_max = self.calc_x2_char(self.alpha_max, t1)
        x1_max = self.calc_x1_char(self.alpha_max, t1, x2_max)
        x2_max_a = self.calc_x2_char(self.alpha_const, t1_max)
        x1_max_a = self.calc_x1_char(self.alpha_const, t1_max, x2_max_a)

        # Medium
        x2_med = self.calc_x2_char(self.alpha_med, t1)
        x1_med = self.calc_x1_char(self.alpha_med, t1, x2_med)
        x2_med_a = self.calc_x2_char(self.alpha_const, t1_med)
        x1_med_a = self.calc_x1_char(self.alpha_const, t1_med, x2_med_a)

        x2_hat_min = self.x2_hat_min - t2_min * g
        x2_hat_max = self.x2_hat_max - t2_max * g
        x2_hat_med = self.x2_hat_med - t2_med * g
        x2_hat_min_a = self.x2_hat_min - t2_min_a * g
        x2_hat_max_a = self.x2_hat_max - t2_max_a * g
        x2_hat_med_a = self.x2_hat_med - t2_med_a * g

        x1_hat_min = self.x1_hat_min - 0.5 * x2_hat_min ** 2 / g
        x1_hat_max = self.x1_hat_max - 0.5 * x2_hat_max ** 2 / g
        x1_hat_med = self.x1_hat_med - 0.5 * x2_hat_med ** 2 / g
        x1_hat_min_a = self.x1_hat_min_a - 0.5 * x2_hat_min_a ** 2 / g
        x1_hat_max_a = self.x1_hat_max_a - 0.5 * x2_hat_max_a ** 2 / g
        x1_hat_med_a = self.x1_hat_med_a - 0.5 * x2_hat_med_a ** 2 / g

        plt.figure()
        plt.title('Total burn time: '+ str(self.t1) + '[s]')
        plt.grid()
        plt.xlabel("Altitude [m]")
        plt.ylabel("Velocity [m/s]")
        plt.plot(x1_hat_min, x2_hat_min, '--k', label='free-fall')
        plt.plot(x1_min, x2_min, 'k', label=r'sf: $\alpha_{min}$ =' + str(round(self.alpha_min, 2)))
        plt.plot(x1_hat_med, x2_hat_med, '--b', label='free-fall')
        plt.plot(x1_med, x2_med, 'b', label=r'sf: $\alpha_{med}$ =' + str(round(self.alpha_med, 2)))
        plt.plot(x1_hat_max, x2_hat_max, '--r', label='free-fall')
        plt.plot(x1_max, x2_max, 'r', label=r'sf: $\alpha_{max}$ =' + str(round(self.alpha_max, 2)))
        plt.legend()

        plt.figure()
        plt.title('Mass flow rate: ' + str(round(self.alpha_const, 2)) + '[kg/s]')
        plt.grid()
        plt.xlabel("Altitude [m]")
        plt.ylabel("Velocity [m/s]")
        plt.plot(x1_hat_min_a, x2_hat_min_a, '--k', label='free-fall')
        plt.plot(x1_min_a, x2_min_a, 'k', label=r'sf: $tb_{min}$ =' + str(round(self.t1_min, 2)))
        plt.plot(x1_hat_med_a, x2_hat_med_a, '--b', label='free-fall')
        plt.plot(x1_med_a, x2_med_a, 'b', label=r'sf: $tb_{med}$ =' + str(round(self.t1_med, 2)))
        plt.plot(x1_hat_max_a, x2_hat_max_a, '--r', label='free-fall')
        plt.plot(x1_max_a, x2_max_a, 'r', label=r'sf: $tb_{max}$ =' + str(round(self.t1_max, 2)))
        plt.legend()
        plt.show()
        return

    def calc_optimal_parameters(self, init_state, max_generation, n_individuals, range_variables):
        ga = GeneticAlgorithm(1, self.g_planet, init_state, max_generation, n_individuals, range_variables)
        ga.optimize(self.rungeonestep, self.c_char)
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
