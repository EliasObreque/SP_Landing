"""
Created by Elias Obreque
Date: 23-11-2023
email: els.obrq@gmail.com
"""
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import random
from core.module.Module import Module
from core.thrust.thrustProperties import main_thruster, second_thruster, third_thruster
from core.thrust.propellant.propellantProperties import main_propellant, second_propellant, third_propellant
from tools.mathtools import propagate_rv_by_ang
from tools.pso import PSOStandard
from tools.Viewer import plot_orbit_solution, plot_state_solution, plot_pso_result

random.seed(10)

mass_0 = 24.0
inertia_0 = 1 / 12 * mass_0 * (0.2 ** 2 + 0.3 ** 2)

mu = 4.9048695e12  # m3s-2
rm = 1.738e6
ra = 68e6
rp = 2e6
a = 0.5 * (ra + rp)
ecc = 1 - rp / a
b = a * np.sqrt(1 - ecc ** 2)
vp = np.sqrt(2 * mu / rp - mu / a)
va = np.sqrt(2 * mu / ra - mu / a)
p_ = rp * (1 + ecc)
f_0 = -np.pi / 2
r_0 = p_ / (1 + ecc * np.cos(f_0))

rot_90 = np.array([[0, -1], [1, 0]])
position = rot_90 @ np.array([np.cos(f_0), np.sin(f_0)]) * r_0
velocity = np.sqrt(mu / p_) * rot_90 @ np.array([-np.sin(f_0), (ecc + np.cos(f_0))])

theta = np.array(270.0 * np.deg2rad(1))
omega = np.array(0.0)
dt = 0.1
tf = min(1500000, 2 * np.pi / np.sqrt(mu) * a ** (3 / 2))
state = [position, velocity, theta, omega]

# TARGET
h_target = rm + 2e3
rp_target = 2e6

energy_target = -mu / h_target

thruster_pos = np.array([[-0.06975, -0.0],  # Main Thruster
                         [-0.06975, -0.0887],  # Second -1- Thruster
                         [-0.06975, 0.0887],  # Second -2- Thruster
                         [-0.06975, -0.0887],  # Second -3- Thruster
                         [-0.06975, 0.0887],  # Second -4- Thruster
                         ])
thruster_pos += np.random.normal(0, 0.0001, size=np.shape(thruster_pos))
thruster_ang = np.random.normal(0, np.deg2rad(0.5), size=(len(thruster_pos)))

thruster_properties_ = [copy.deepcopy(main_thruster),  # Main Thruster
                        copy.deepcopy(second_thruster),  # Second -1- Thruster
                        copy.deepcopy(second_thruster),  # Second -2- Thruster
                        copy.deepcopy(second_thruster),  # Second -3- Thruster
                        copy.deepcopy(second_thruster),  # Second -4- Thruster
                        ]
propellant_properties_ = [copy.deepcopy(main_propellant),  # Main Thruster
                          copy.deepcopy(second_propellant),  # Second -1- Thruster
                          copy.deepcopy(second_propellant),  # Second -2- Thruster
                          copy.deepcopy(second_propellant),  # Second -3- Thruster
                          copy.deepcopy(second_propellant),  # Second -4- Thruster
                          ]

noise_state = [np.random.normal(0, 1e3, size=2), np.random.normal(0, 5, size=2)]


def descent_optimization(modules_setting_):
    control_set_ = [modules_setting_[0],
                    # modules_setting_[1], modules_setting_[1],
                    # modules_setting_[2], modules_setting_[2],
                    ]
    state_ = [state[0] + noise_state[0],
              state[1] + noise_state[1],
              state[2],
              state[3]]

    # transform cartesian r and v into perifocal r and v
    sorted_ignition = sorted(modules_setting_)
    mass_, inertia_ = mass_0, inertia_0
    for i in range(len(sorted_ignition)):
        idx = np.argwhere(modules_setting_ == sorted_ignition[i])[0][0]
        if idx == 0:
            # FIRST
            state_[0], state_[1] = propagate_rv_by_ang(state_[0], state_[1], modules_setting_[idx], mu)
            module = Module(mass_, inertia_, state_,
                            [thruster_pos[0]], [thruster_ang[0]], [thruster_properties_[0]],
                            [propellant_properties_[0]], "2D", dt, training=True)

            module.set_control_function([control_set_[0]])
            historical_state = module.simulate(tf, low_step=0.1, progress=False, only_thrust=True)
            r_, v_ = historical_state[0][-1], historical_state[1][-1]
            state_ = [r_,
                      v_,
                      historical_state[3][-1],
                      historical_state[4][-1]]
            mass_, inertia_ = historical_state[2][-1], historical_state[5][-1]
        elif idx == 1:
            # SECOND
            state_[0], state_[1] = propagate_rv_by_ang(state_[0], state_[1], modules_setting_[idx], mu)
            module = Module(mass_, inertia_, state_,
                            thruster_pos[1:3], thruster_ang[1:3], thruster_properties_[1:3],
                            propellant_properties_[1:3], "2D", dt, training=True)

            module.set_control_function(control_set_[1:3])
            historical_state = module.simulate(tf, low_step=0.1, progress=False, only_thrust=True)
            r_, v_ = historical_state[0][-1], historical_state[1][-1]
            state_ = [r_,
                      v_,
                      historical_state[3][-1],
                      historical_state[4][-1]]
            mass_, inertia_ = historical_state[2][-1], historical_state[5][-1]
        elif idx == 2:
            # THIRD
            state_[0], state_[1] = propagate_rv_by_ang(state_[0], state_[1], modules_setting_[idx], mu)
            module = Module(mass_, inertia_, state_,
                            thruster_pos[3:], thruster_ang[3:], thruster_properties_[3:],
                            propellant_properties_[3:], "2D", dt, training=True)

            module.set_control_function(control_set_[3:])
            historical_state = module.simulate(tf, low_step=0.1, progress=False, only_thrust=True)
            r_, v_ = historical_state[0][-1], historical_state[1][-1]
            state_ = [r_,
                      v_,
                      historical_state[3][-1],
                      historical_state[4][-1]]
            mass_, inertia_ = historical_state[2][-1], historical_state[5][-1]
    # COST
    r_state = np.array([np.linalg.norm(elem) for elem in historical_state[0]])
    v_state = np.array([np.linalg.norm(elem) for elem in historical_state[1]])
    mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
    state_energy = historical_state[8]

    # error = (state_energy[-1] - energy_target) ** 2 / mass_state[-1]

    ang = np.arctan2(historical_state[0][-1][1], historical_state[0][-1][0])
    v_t_n = np.array([[np.cos(ang - np.pi / 2), -np.sin(ang - np.pi / 2)],
                      [np.sin(ang - np.pi / 2), np.cos(ang - np.pi / 2)]]).T @ historical_state[1][-1]

    error = state_energy[-1] + 100 * v_t_n[0] ** 2
    # error += 1000 if module.dynamics.isTouchdown() else 1
    return error, historical_state


if __name__ == '__main__':

    n_step = 200
    n_par = 5
    folder = "logs/"
    name = "descent_ignition_3"
    stage = "D"
    plot_flag = True

    range_variables = [(0.0, 2 * np.pi),  # First ignition angular (rad)
                       # (0.0, 2 * np.pi),  # Second -1- ignition angular (rad)
                       # (0.0, 2 * np.pi),  # Second -2- ignition angular (rad)
                       ]

    pso_algorithm = PSOStandard(descent_optimization, n_particles=n_par, n_steps=n_step)
    pso_algorithm.initialize(range_variables)

    init_time = time.time()
    final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_algorithm.optimize(clip=True)
    end_time = time.time()
    print("Optimization Time: {}".format((end_time - init_time) / 60))
    modules_setting = pso_algorithm.gbest_position
    sorted_ignition = sorted(modules_setting)

    state = [position, velocity, theta, omega]
    state_ = [state[0] + noise_state[0],
              state[1] + noise_state[1],
              state[2],
              state[3]]

    module = Module(mass_0, inertia_0, state_,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)
    control_set_ = [modules_setting[0],
                    # modules_setting[1], modules_setting[1],
                    # modules_setting[2], modules_setting[2],
                    ]
    module.set_control_function([control_set_])
    historical_state = module.simulate(30000, low_step=0.01, progress=False)
    list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                 "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]

    if plot_flag:
        plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name, plot_flag=plot_flag)
        plot_state_solution(historical_state, list_name, folder, name, aux={8: energy_target},
                            plot_flag=plot_flag)
        plot_orbit_solution([historical_state], ["orbit"], a, b, rp, folder, name,
                            h_target=h_target, plot_flag=plot_flag)
        plt.show(block=True)