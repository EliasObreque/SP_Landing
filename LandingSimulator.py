"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle

from core.module.Module import Module
from core.thrust.thrustProperties import default_thruster
from core.thrust.propellant.propellantProperties import default_propellant
from tools.pso import PSOStandard
from tools.Viewer import plot_orbit_solution, plot_state_solution

mass_0 = 24.0
inertia_0 = 100
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
theta = 270 * np.deg2rad(1)
q_i2b = np.array([0, 0, 0, 1])
omega = 0

# TARGET
h_target = rm + 100e3
rp_target = 2e6
energy_target = [-mu / (rp_target + h_target),
                 -mu / (2 * h_target)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, help="Folder name")
    parser.add_argument("-n", "--name", type=str, help="Files head name")
    parser.add_argument("-bs", "--batch", type=int, help="Batch Size")
    parser.add_argument("-l", "--loop", type=int, help="Number of loops")
    parser.add_argument("-s", "--stage", type=str, help="Stage, Descend (D), Landing (L)")
    parser.add_argument("-ps", "--plot_show", action="store_false", help="Show")
    return parser.parse_args()


def get_energy(mu, r, v):
    return 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)


def cost_function_descend(modules_setting_):
    state = [position, velocity, theta, omega]
    dt = 0.1
    tf = 1210000

    n_thrusters = 2
    thruster_pos = [np.array([0, 0])] * 2
    thruster_ang = [0] * 2
    thr_properties = copy.deepcopy(default_thruster)
    prp_properties = copy.deepcopy(default_propellant)

    thruster_properties_ = [copy.deepcopy(thr_properties) for _ in range(len(modules_setting_[1::2]))]
    propellant_properties_ = [copy.deepcopy(prp_properties) for _ in range(len(modules_setting_[1::2]))]

    thruster_properties_[1]['throat_diameter'] = 0.005
    propellant_properties_[1]['geometry']['setting']['int_diameter'] = 0.01

    state_ = [state[0] + np.random.normal(0, 100, size=2),
              state[1] + np.random.normal(0, 5, size=2),
              state[2],
              state[3]]
    modules_ = Module(mass_0, inertia_0, state_,
                      thruster_pos, thruster_ang, thruster_properties_,
                      propellant_properties_, "2D", dt, training=True)

    engine_diameter = modules_setting[1::2]
    control_set_ = modules_setting[0::2]
    modules_.set_thrust_design(engine_diameter, 0)
    modules_.set_control_function(control_set_)
    historical_state = modules_.simulate(tf, low_step=0.01, progress=False)
    r_state = np.array([np.linalg.norm(elem) for elem in historical_state[0]])
    mass_state = np.array([np.linalg.norm(elem) for elem in historical_state[2]])
    state_energy = historical_state[8]
    error = 0
    for j, act in enumerate(modules_.thrusters_action_wind):
        if len(act) > 0:
            error += np.abs(state_energy[min(act[1] + 1, len(state_energy) - 5)] - energy_target[j]) / mass_state[-1]
        else:
            error += np.abs(state_energy[-1] - energy_target[j]) / mass_state[-1]
    error *= 1000 if modules_.dynamics.isTouchdown() else 1
    error *= 100 if modules_.dynamics.notMass() else 1
    error *= 1000 if min(r_state) < h_target * 0.9 else 1

    modules_.reset()
    return error, historical_state


if __name__ == '__main__':
    n_step = 50
    n_par = 30
    folder = "logs/"
    name = "test"
    stage = "D"
    plot_flag = True

    batch_size = 50
    n_loop = 5

    args = parse_args()

    if args.folder:
        folder = folder + args.folder + "/"
        # create folder
        if not os.path.exists(os.path.normpath(folder)):
            os.makedirs(os.path.normpath(folder))
    else:
        folder = folder + "plane/"

    if args.name:
        name = args.name

    if args.batch:
        batch_size = args.batch

    if args.loop:
        n_loop = args.loop

    if args.stage:
        stage = args.stage

    plot_flag = args.plot_show

    for nl in range(n_loop):
        for nb in range(batch_size):
            if stage == "D":
                # Optimal Design of the Control
                # (First stage: Decrease the altitude, and the mass to decrease the rw mass/inertia)
                range_variables = [(1.0, 2),  # First ignition position (angle)
                                   (0.1, 0.19),  # Main engine diameter (meter)
                                   (3, 6),  # Second ignition position (meter)
                                   (0.001, 0.1)  # Secondary engine diameter (meter)
                                   ]

                pso_algorithm = PSOStandard(cost_function_descend, n_particles=n_par, n_steps=n_step)
                pso_algorithm.initialize(range_variables)

                init_time = time.time()
                final_eval, best_state = pso_algorithm.optimize()
                end_time = time.time()
                print("Optimization Time: {}".format((end_time - init_time) / 60))
                pso_algorithm.plot_result(folder, name)

                list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                             "Inertia [kgm2]", "Thrust [N]", "Torque [Nm]", "Energy [J]"]
                plot_state_solution(best_state, list_name, folder, name, aux={8: energy_target})
                plot_orbit_solution([best_state], ["orbit"], folder, name)
                modules_setting = pso_algorithm.gbest_position
            elif stage == "L":
                pass

            if plot_flag:
                plt.show()
