"""
Created by:

@author: Elias Obreque
@Date: 1/6/2021 5:33 PM 
els.obrq@gmail.com

ref: Meditch. J. On the Problem of Optimal Thrust Programming For a Lunar Soft Landing
"""
import numpy as np
from matplotlib import pyplot as plt
tolerance = 1e-9
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'


class HamilCalcLimit(object):
    def __init__(self, mass, c_char, g_planet):
        self.mass = mass
        self.c_char = c_char
        self.g_planet = g_planet
        self.t1 = 0
        self.alpha = 0
        self.b = 0
        self.a = 0
        # Variables for constant time
        self.t2_min_t = 0
        self.t2_max_t = 0
        self.t2_med_t = 0
        self.x2_hat_min_t = 0
        self.x2_hat_max_t = 0
        self.x2_hat_med_t = 0
        self.x1_hat_min_t = 0
        self.x1_hat_max_t = 0
        self.x1_hat_med_t = 0
        # Variables for constant alpha
        self.t2_min_a = 0
        self.t2_max_a = 0
        self.t2_med_a = 0
        self.x2_hat_min_a = 0
        self.x2_hat_max_a = 0
        self.x2_hat_med_a = 0
        self.x1_hat_min_a = 0
        self.x1_hat_max_a = 0
        self.x1_hat_med_a = 0
        return

    def calc_limits_with_const_time(self, t1, alpha_min, alpha_max):
        g       = self.g_planet
        self.t1 = t1
        alpha_med = (alpha_max + alpha_min) * 0.5
        # Minimum
        x2_char_min = self.calc_x2_char(alpha_min, t1)
        x1_char_min = self.calc_x1_char(alpha_min, t1,  x2_char_min)
        # Maximum
        x2_char_max = self.calc_x2_char(alpha_max,  t1)
        x1_char_max = self.calc_x1_char(alpha_max,  t1, x2_char_max)
        # Medium
        x2_char_med = self.calc_x2_char(alpha_med,  t1)
        x1_char_med = self.calc_x1_char(alpha_med,  t1, x2_char_med)

        print("Minimum state parameters *")
        print(x2_char_min, " [m/s] - ",  x1_char_min, " [m]\n")
        print("Medium state parameters * ")
        print(x2_char_med, " [m/s] - ",  x1_char_med, " [m]\n")
        print("Maximum state parameters * ")
        print(x2_char_max, " [m/s] - ",   x1_char_max, " [m]\n")

        self.t2_min_t = x2_char_min / g
        self.t2_max_t = x2_char_max / g
        self.t2_med_t = x2_char_med / g
        self.x1_hat_min_t = x1_char_min - 0.5 * x2_char_min ** 2 / g
        self.x1_hat_max_t = x1_char_max - 0.5 * x2_char_max ** 2 / g
        self.x1_hat_med_t = x1_char_med - 0.5 * x2_char_med ** 2 / g
        print("Initial state [min, med, max] [m]")
        print("[",  self.x1_hat_min_t, " ,",  self.x1_hat_med_t, " ,",  self.x1_hat_max_t, "]")
        return

    def calc_limits_with_const_alpha(self, t1_min, t1_max, alpha):
        self.alpha = alpha
        g = self.g_planet
        t1_med = 0.5 * (t1_max + t1_min)
        # Minimum
        x2_char_min_a = self.calc_x2_char(alpha,  t1_min)
        x1_char_min_a = self.calc_x1_char(alpha,  t1_min, x2_char_min_a)
        # Maximum
        x2_char_max_a = self.calc_x2_char(alpha,  t1_max)
        x1_char_max_a = self.calc_x1_char(alpha,  t1_max, x2_char_max_a)
        # Medium
        x2_char_med_a = self.calc_x2_char(alpha, t1_med)
        x1_char_med_a = self.calc_x1_char(alpha,  t1_med, x2_char_med_a)

        print("Minimum state parameters *")
        print(x2_char_min_a, " [m/s] - ",  x1_char_min_a, " [m]\n")
        print("Medium state parameters * ")
        print(x2_char_med_a, " [m/s] - ",  x1_char_med_a, " [m]\n")
        print("Maximum state parameters * ")
        print(x2_char_max_a, " [m/s] - ",  x1_char_max_a, " [m]\n")

        self.t2_min_a = x2_char_min_a / g
        self.t2_max_a = x2_char_max_a / g
        self.t2_med_a = x2_char_med_a / g
        self.x1_hat_min_a = x1_char_min_a - 0.5 * x2_char_min_a ** 2 / g
        self.x1_hat_max_a = x1_char_max_a - 0.5 * x2_char_max_a ** 2 / g
        self.x1_hat_med_a = x1_char_med_a - 0.5 * x2_char_med_a ** 2 / g
        print("Initial state [min, med, max] [m]")
        print("[", self.x1_hat_min_a, " ,", self.x1_hat_med_a, " ,", self.x1_hat_max_a, "]")
        return

    def calc_x2_char(self, alpha, t1):
        g = self.g_planet
        arg_ = 1 - alpha * t1 / self.mass
        return self.c_char * np.log(arg_) - g * t1

    def calc_x1_char(self, alpha, t1, x2_c=None):
        g = self.g_planet
        arg_ = 1 - alpha * t1 / self.mass
        return 0.5 * g * t1 ** 2 - self.c_char * t1 - self.c_char * self.mass * np.log(arg_) / alpha

    def calc_simple_optimal_parameters(self, r0, alpha_min, alpha_max, t_burn):
        a_cur = alpha_min
        b_cur = alpha_max
        f_l, f_r = self.bisection_method(a_cur, b_cur, r0, t_burn)
        c_cur = (a_cur + b_cur)/2
        f_c, _ = self.bisection_method(c_cur, 0, r0, t_burn)
        c_last = c_cur
        while np.abs(f_c) > tolerance:
            if f_c < 0 and f_l < 0:
                a_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur, r0, t_burn)
            elif f_c < 0 < f_l:
                b_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur, r0, t_burn)
            elif f_c > 0 and f_r > 0:
                b_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur, r0, t_burn)
            else:
                a_cur = c_last
                f_l, f_r = self.bisection_method(a_cur, b_cur, r0, t_burn)
            c_cur = (a_cur + b_cur)/2
            f_c, _ = self.bisection_method(c_cur, 0, r0, t_burn)
            c_last = c_cur
        self.alpha = c_last
        self.b = self.c_char * self.alpha ** 2 / (2 * self.mass ** 2)
        self.a = 0.5 * (self.c_char * self.alpha + self.g_planet * self.mass) / self.mass
        print('--------------------------------------------------------------------------')
        print("Optimal alpha (m_dot) for t_burn = ", t_burn, " [s]: ", self.alpha)
        print('Parameters [a] and [b]: ', self.a, ' - ', self.b)
        print('Height error: ', f_c)
        return self.alpha

    def calc_parameters(self):
        self.b = self.c_char * self.alpha ** 2 / (2 * self.mass ** 2)
        self.a = 0.5 * (self.c_char * self.alpha + self.g_planet * self.mass) / self.mass
        return

    def bisection_method(self, var_left, var_right, r0, t_burn):
        f_l = r0 - self.calc_x1_char(var_left, t_burn) +\
              0.5 * self.calc_x2_char(var_left, t_burn) ** 2 / self.g_planet
        if var_right != 0:
            f_r = r0 - self.calc_x1_char(var_right, t_burn) +\
                  0.5 * self.calc_x2_char(var_right, t_burn) ** 2 / self.g_planet
        else:
            f_r = 0
        return -f_l, -f_r

    def calc_first_cond(self, alt, vel, alpha):
        return np.abs(vel) - alt * alpha / self.mass

    def get_signal_control(self, state):
        x1 = state[0]
        x2 = state[1]
        if x1 >= 0:
            sf = (self.b / self.a) * x1 + 2 * self.a * np.sqrt(x1 / self.a) + x2
        else:
            sf = 0
        return sf

    @staticmethod
    def print_simulation_data(x_states, mp, m0, r0):
        print('--------------------------------------------------------------------------')
        print('Error position [m]: ', abs(round(x_states[-1, 0], 2)),
              ' - Error velocity [m/s]: ', abs(round(x_states[-1, 1], 2)),
              'Final mass [kg]: ', round(x_states[-1, 2], 2))
        print('Error position %: ', abs(round(x_states[-1, 0] / r0 * 100.0, 2)),
              'Used mass [kg]: ', round(m0 - x_states[-1, 2], 2), '[', round(x_states[-1, 2] / m0 * 100.0, 2), ' %]',
              'Theoretical mass [kg]:', mp)
        return

    def calc_online_t1(self, x1, x2, x3, alpha):
        A = -0.5*self.g_planet * alpha
        B = (self.c_char * alpha + self.g_planet * x3)
        C = x1 * alpha + x2 * x3
        t1 = (-B + np.sqrt(B**2 - 4*A*C))/(2 * A)
        return t1

    def show_alpha_limits(self, alpha_min, alpha_max):
        alpha_med = 0.5 * (alpha_min + alpha_max)
        g = self.g_planet

        t1 = np.linspace(0, self.t1, 200)
        t2_min = np.linspace(0, self.t2_min_t)
        t2_max = np.linspace(0, self.t2_max_t)
        t2_med = np.linspace(0, self.t2_med_t)

        # Minimum
        x2_min = self.calc_x2_char(alpha_min, t1)
        x1_min = self.calc_x1_char(alpha_min, t1, x2_min)

        # Maximum
        x2_max = self.calc_x2_char(alpha_max, t1)
        x1_max = self.calc_x1_char(alpha_max, t1, x2_max)

        # Medium
        x2_med = self.calc_x2_char(alpha_med, t1)
        x1_med = self.calc_x1_char(alpha_med, t1, x2_med)

        x2_hat_min = self.x2_hat_min_t + t2_min * g
        x2_hat_max = self.x2_hat_max_t + t2_max * g
        x2_hat_med = self.x2_hat_med_t + t2_med * g

        x1_hat_min = self.x1_hat_min_t + 0.5 * x2_hat_min ** 2 / g
        x1_hat_max = self.x1_hat_max_t + 0.5 * x2_hat_max ** 2 / g
        x1_hat_med = self.x1_hat_med_t + 0.5 * x2_hat_med ** 2 / g

        plt.figure()
        plt.title('Total burn time: ' + str(self.t1) + '[s]')
        plt.grid()
        plt.xlabel("Altitude [m]")
        plt.ylabel("Velocity [m/s]")
        plt.plot(x1_hat_min, x2_hat_min, '--k', label='free-fall')
        plt.plot(x1_min, x2_min, 'k', label=r'sf: $\alpha_{min}$ =' + str(round(alpha_min, 2)))
        plt.plot(x1_hat_med, x2_hat_med, '--b', label='free-fall')
        plt.plot(x1_med, x2_med, 'b', label=r'sf: $\alpha_{med}$ =' + str(round(alpha_med, 2)))
        plt.plot(x1_hat_max, x2_hat_max, '--r', label='free-fall')
        plt.plot(x1_max, x2_max, 'r', label=r'sf: $\alpha_{max}$ =' + str(round(alpha_max, 2)))
        plt.legend()
        plt.show(block=False)
        return

    def show_time_limits(self, t_burn_min, t_burn_max):
        t_burn_med = 0.5 * (t_burn_min + t_burn_max)
        g = self.g_planet

        t1_max = np.linspace(0, t_burn_max, 200)
        t1_med = np.linspace(0, t_burn_med, 200)
        t1_min = np.linspace(0, t_burn_min, 200)

        t2_min_a = np.linspace(0, self.t2_min_a)
        t2_max_a = np.linspace(0, self.t2_max_a)
        t2_med_a = np.linspace(0, self.t2_med_a)

        # Minimum
        x2_min_a = self.calc_x2_char(self.alpha, t1_min)
        x1_min_a = self.calc_x1_char(self.alpha, t1_min, x2_min_a)

        # Maximum
        x2_max_a = self.calc_x2_char(self.alpha, t1_max)
        x1_max_a = self.calc_x1_char(self.alpha, t1_max, x2_max_a)

        # Medium
        x2_med_a = self.calc_x2_char(self.alpha, t1_med)
        x1_med_a = self.calc_x1_char(self.alpha, t1_med, x2_med_a)

        x2_hat_min_a = self.x2_hat_min_t + t2_min_a * g
        x2_hat_max_a = self.x2_hat_max_t + t2_max_a * g
        x2_hat_med_a = self.x2_hat_med_t + t2_med_a * g

        x1_hat_min_a = self.x1_hat_min_a + 0.5 * x2_hat_min_a ** 2 / g
        x1_hat_max_a = self.x1_hat_max_a + 0.5 * x2_hat_max_a ** 2 / g
        x1_hat_med_a = self.x1_hat_med_a + 0.5 * x2_hat_med_a ** 2 / g

        plt.figure()
        plt.title('Mass flow rate: ' + str(round(self.alpha, 2)) + '[kg/s]')
        plt.grid()
        plt.xlabel("Altitude [m]")
        plt.ylabel("Velocity [m/s]")
        plt.plot(x1_hat_min_a, x2_hat_min_a, '--k', label='free-fall')
        plt.plot(x1_min_a, x2_min_a, 'k', label=r'sf: $tb_{min}$ =' + str(round(t_burn_min, 2)))
        plt.plot(x1_hat_med_a, x2_hat_med_a, '--b', label='free-fall')
        plt.plot(x1_med_a, x2_med_a, 'b', label=r'sf: $tb_{med}$ =' + str(round(t_burn_med, 2)))
        plt.plot(x1_hat_max_a, x2_hat_max_a, '--r', label='free-fall')
        plt.plot(x1_max_a, x2_max_a, 'r', label=r'sf: $tb_{max}$ =' + str(round(t_burn_max, 2)))
        plt.legend()
        plt.show(block=False)
        return

    @staticmethod
    def plot_1d_simulation(x_states, time_series, thr):
        """
        Comments:
        - The graphs show a decrease close to the optimum
        - The relative error is low at the endpoint
        - Although the error is small, in practice this solution is not viable because it does not consider the
         constraint x1> 0.
        """
        plt.figure()
        plt.grid()
        plt.ylabel('Altitude [m]')
        plt.xlabel('Velocity [m/s]')
        plt.plot(x_states[:, 1], x_states[:, 0])

        plt.figure()
        plt.ylabel('Altitude [m]')
        plt.xlabel('Time [s]')
        plt.grid()
        plt.plot(time_series, x_states[:, 0])

        plt.figure()
        plt.ylabel('Velocity [m/s]')
        plt.xlabel('Time [s]')
        plt.grid()
        plt.plot(time_series, x_states[:, 1])

        plt.figure()
        plt.ylabel('Thrust [N]')
        plt.xlabel('Time [s]')
        plt.grid()
        plt.plot(time_series, thr)
