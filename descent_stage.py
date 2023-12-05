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
from core.thrust.thrustProperties import main_thruster
from core.thrust.propellant.propellantProperties import main_propellant
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
                         ])
thruster_pos += np.random.normal(0, 0.0001, size=np.shape(thruster_pos))
thruster_ang = np.random.normal(0, np.deg2rad(0.1), size=(len(thruster_pos)))

thruster_properties_ = [copy.deepcopy(main_thruster),  # Main Thruster
                        ]
propellant_properties_ = [copy.deepcopy(main_propellant),  # Main Thruster
                          ]

list_name = ["Position [m]", "Velocity [km/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
             "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]
list_gain = [-1]
folder = "logs/progressive/train/"
name_ = "mass_opt_vf_"
sigma_r = 50
sigma_v = 5


def descent_optimization(modules_setting_):
    control_set_ = modules_setting_[:3]

    state_ = [state[0],
              state[1],
              state[2],
              state[3]]
    mass_, inertia_ = mass_0, inertia_0

    module = Module(mass_, inertia_, state_, sigma_r, sigma_v,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)

    module.set_control_function([control_set_[0], control_set_[1], control_set_[1], control_set_[2], control_set_[2]])

    module.set_thrust_design([modules_setting_[3],
                              modules_setting_[4], modules_setting_[4],
                              modules_setting_[5], modules_setting_[5]],
                             [modules_setting_[6],
                              modules_setting_[7], modules_setting_[7],
                              modules_setting_[8], modules_setting_[8]])

    historical_state = module.simulate(tf, low_step=0.1, progress=False)
    mass_, inertia_ = historical_state[2][-1], historical_state[5][-1]

    # COST
    r_state = np.array([np.linalg.norm(elem) for elem in historical_state[0]])
    # v_state = np.array([np.linalg.norm(elem) for elem in historical_state[1]])
    mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
    # state_energy = historical_state[8]

    ang = np.arctan2(historical_state[0][-1][1], historical_state[0][-1][0])
    v_t_n = np.array([[np.cos(ang - np.pi / 2), -np.sin(ang - np.pi / 2)],
                      [np.sin(ang - np.pi / 2), np.cos(ang - np.pi / 2)]]).T @ historical_state[1][-1]

    error = r_state[-1] * 1e-3 + np.sqrt(10000 * v_t_n[0] ** 2 + v_t_n[1] ** 2)

    return error, historical_state


if __name__ == '__main__':

    n_step = 50
    n_par = 40
    folder = "logs/"
    name = "descent_ignition_3"
    stage = "D"
    plot_flag = True

    range_variables = [(10, 300),  # First ignition angular (altitude km)
                       (1.2 * np.pi, 1.8 * np.pi),  # Second -1- ignition angular (rad)
                       (1.2 * np.pi, 1.8 * np.pi),  # Second -2- ignition angular (rad)
                       (0.1, 0.15),  # First Diameter
                       (0.05, 0.06),  # Second Diameter
                       (0.05, 0.06),  # Third Diameter
                       (0.2, 0.2),  # First Large
                       (0.1, 0.2),  # Second Large
                       (0.1, 0.2)  # Third Large
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
    state_ = [state[0],
              state[1],
              state[2],
              state[3]]

    module = Module(mass_0, inertia_0, state_,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)
    control_set_ = modules_setting[:3]
    module.set_control_function(control_set_)
    module.set_thrust_design(modules_setting[3:6], modules_setting[6:9])

    historical_state = module.simulate(15000000, low_step=0.1, progress=False)
    list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                 "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]
    historical_state[1] = np.array(historical_state[1]) / 1000
    if plot_flag:
        plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name, plot_flag=plot_flag)
        plot_state_solution(historical_state, list_name, folder, name, aux={8: energy_target},
                            plot_flag=plot_flag)
        plot_orbit_solution([historical_state], ["orbit"], a, b, rp, folder, name,
                            h_target=h_target, plot_flag=plot_flag)
        plt.show(block=True)