"""
Created by Elias Obreque
Date: 14-06-2023
email: els.obrq@gmail.com
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from pso import PSOStandard
import matplotlib.pyplot as plt
from tools.mathtools import runge_kutta_4

g = 1.67
c_sp = 250 * 9.8
burn_time = 10
THRUST_LEVEL = 350
dt = 0.01
count_thrust = int(burn_time / dt)
c_count = 0


class ThrustModel:
    def __init__(self, dt, level):
        self.burn_time = 5
        self.dt = dt
        self.current_time = 0
        self.thrust = 0
        self.max_thrust = level
        self.is_burning = False
        self.was_burned = False
        self.historical = [0]

    def get_thrust(self, control_signal):
        self.thrust = 0
        if control_signal == 1 and not self.was_burned:
            self.is_burning = True
        if self.is_burning:
            self.thrust = self.max_thrust - self.max_thrust * 0.5 * self.current_time / self.burn_time
            # self.thrust = self.max_thrust * np.sin(self.current_time/self.burn_time * np.pi)
            self.current_time += self.dt
        if self.current_time > self.burn_time:
            self.is_burning = False
            self.was_burned = True
        return self.thrust

    def save(self):
        self.historical.append(self.thrust)

    def get_historical(self):
        return np.array(self.historical)


def model_thrust(mass, l2, l3):
    sf = l2 / mass - l3 / c_sp
    if isinstance(sf, float):
        ignition = 1 if sf < 0 else 0
    else:
        ignition = 0
    return ignition


def dynamic_norm(tau, x, p):
    tf = p[0]
    lambda_10 = p[1]

    y, v, m, lambda_2, lambda_3 = x[0], x[1], x[2], x[3], x[4]
    dy = np.zeros((5, len(tau)))
    dc_costate = np.zeros(len(tau))
    # thrust_model = ThrustModel(dt, 405)

    sig = [model_thrust(m_, l2_, l3_) for m_, l2_, l3_ in zip(m, lambda_2, lambda_3)]

    thrust = np.asarray([500 if sig_ else 0 for sig_ in sig])  # thrust_model.get_thrust(sig)

    dy[0] = v
    dy[1] = (thrust / m / 24 - g) / 2000
    dy[2] = - thrust / c_sp / 24
    dy[3] = - lambda_10
    dy[4] = thrust * lambda_2 / m ** 2 / 24
    dy[0][m < 0.01] = 0
    dy[1][m < 0.01] = 0
    dy[2][m < 0.01] = 0
    return dy * tf


def dynamic_ham(x, *args):
    lambda_10 = 1
    lambda_20 = args[0][0]
    lambda_2 = lambda_20 - lambda_10 * args[1]

    y, v, m = x[0], x[1], x[2]
    thrust = args[2]

    dy = np.zeros(4)
    dy[0] = v
    dy[1] = (thrust / m - g)
    dy[2] = -thrust / c_sp
    dy[3] = thrust * lambda_2 / m ** 2
    return dy


def dynamic(x, *args):
    y, v, m = x[0], x[1], x[2]

    dy = np.zeros(3)
    thrust = args[0]
    dy[0] = v
    dy[1] = (thrust / m - g)
    dy[2] = -thrust / c_sp
    return dy


def cost_func_ham(particle_sol):
    ct = 0
    y_init = [2000, 0, 24, particle_sol[1]]
    y_state = y_init
    end_condition = False
    lambda_10 = 1
    h_ = [lambda_10 * y_state[1] - particle_sol[0] * g + (particle_sol[0] / y_state[2] - y_state[-1] / c_sp) * 0]
    historical = [y_state]
    lambda_2 = particle_sol[0] - 1 * ct
    sf_list = [(lambda_2 / y_state[2] - y_state[-1] / c_sp)]
    c_count = 0
    thrust_model = ThrustModel(dt, particle_sol[2])
    time_array = [0]
    while not end_condition:
        lambda_2 = particle_sol[0] - 1 * ct
        sig = model_thrust(y_state[2], lambda_2, y_state[3])
        new_state = y_state + runge_kutta_4(dynamic_ham, y_state, dt, particle_sol, ct, thrust_model.get_thrust(sig))
        y_state = new_state
        ct += dt
        sf_ = ((particle_sol[0] - lambda_10 * ct) / y_state[2] - y_state[-1] / c_sp)
        sf_list.append(sf_)
        ignition = particle_sol[2] if sf_ < 0 else 0
        h_i = lambda_10 * y_state[1] - (particle_sol[0] - lambda_10 * ct) * g + sf_ * ignition
        h_.append(h_i)
        historical.append(y_state)
        thrust_model.save()
        time_array.append(ct)
        if (y_state[0] < 0.0 or y_state[2] < y_state[2] * 0.1) and thrust_model.was_burned:
            end_condition = True

    historical = np.concatenate([np.array(time_array).reshape(-1, 1).T,
                                 np.array(historical).T,
                                 thrust_model.get_historical().reshape(-1, 1).T,
                                 np.array(h_).reshape(-1, 1).T,
                                 np.array(sf_list).reshape(-1, 1).T]).T

    error = np.sqrt(y_state[0] ** 2 + y_state[1] ** 2 + 0.5 * (y_state[3]) ** 2)
    if y_state[0] < -0.01:
        error *= 10
    if y_state[2] < 0:
        error *= 10
    return error, historical


def cost_func(particle_sol):
    ct = 0
    y_init = [2000, 0, 24]
    y_state = y_init
    t_free = (y_init[1] + np.sqrt(y_init[1]**2 + 4 * g * 0.5 * y_init[0])) / (2 * g * 0.5)
    end_condition = False
    historical = [np.array(y_state)]
    time_array = [0]
    c_count = 0
    thrust_model = ThrustModel(dt, particle_sol[2])
    while not end_condition:
        sig = 1 if y_state[0] * particle_sol[0] + y_state[1] * particle_sol[1] < 0 else 0
        new_state = y_state + runge_kutta_4(dynamic, y_state, dt, thrust_model.get_thrust(sig), ct)
        y_state = new_state
        ct += dt
        historical.append(y_state)
        thrust_model.save()
        time_array.append(ct)
        if (y_state[0] < 0.0 or y_state[2] < y_state[2] * 0.1) and thrust_model.was_burned:
            end_condition = True

    historical = np.concatenate([np.array(time_array).reshape(-1, 1).T,
                                 np.array(historical).T,
                                 thrust_model.get_historical().reshape(-1, 1).T]).T
    error = y_state[0] ** 2 + y_state[1] ** 2 + (ct / t_free) ** 2
    if y_state[0] < -0.01:
        error *= 10
    if y_state[2] < 0:
        error *= 10
    return error, historical


if __name__ == '__main__':
    y_init = [2000, 0, 24, 1, 1]
    n_par = 50
    n_step = 300

    pso_opt = PSOStandard(name="direct_pso", func=cost_func, n_particles=n_par, n_steps=n_step)
    pso_opt.initialize([[0.0, 1.0],
                        [0, y_init[0] / np.sqrt(y_init[1] ** 2 + y_init[0] * g * 2)],
                        [200, 500]])
    final_eval_opt, min_opt = pso_opt.optimize(clip=False)
    pso_opt.plot_result(min_opt, list_name=["Altitude [m]", 'Velocity [m/s]', 'Mass [kg]', "Thrust [N]"],
                        folder="./res/time_thrust/")

    # pso_haml = PSOStandard(name="hamilton_pso", func=cost_func_ham, n_particles=n_par, n_steps=n_step)
    # pso_haml.initialize([[-50, 50],
    #                      [-100.0, 100.0],
    #                      [400, 500]])
    #
    # final_eval_ham, min_opt_ham = pso_haml.optimize(clip=False)
    # pso_haml.plot_result(min_opt_ham, list_name=["Altitude [m]", 'Velocity [m/s]', 'Mass [kg]', "Co-state [-]",
    #                                              "Thrust [N]", "Hamilton [-]", "Switching function [-]"],
    #                      folder="./res/time_thrust/")
    plt.show()
