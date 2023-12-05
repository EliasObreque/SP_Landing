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
from tools.Viewer import plot_orbit_solution, plot_state_solution, plot_pso_result, plot_general_solution, plot_normal_tangent_velocity
import pickle


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
folder = "logs/attitude/"
name_ = "control_"
sigma_r = 50
sigma_v = 5


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

    state = [position, velocity, theta, omega]
    state_ = [state[0],
              state[1],
              state[2],
              state[3]]

    module = Module(mass_0, inertia_0, state_, sigma_r, sigma_v,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)
    module.set_control_function([2 * np.pi])
    historical_state = module.simulate(500, low_step=0.1, progress=False, force_step=True)
    list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [rad]", "Angular velocity [rad/s]",
                 "Inertia [kgm2]", "Thrust [N]", "Torque [mNm]", "Energy [J]", "Angle Error [rad]",
                 "Angular velocity Error [rad/s]", "RW velocity [rad/s]",
                 "RW Torque [mNm]"]

    historical_state.insert(-1, module.historical_theta_error)
    historical_state.insert(-1, module.historical_omega_error)
    historical_state.insert(-1, module.rw_model.historical_rw_velocity)
    historical_state.insert(-1, np.array(module.rw_model.historical_rw_torque) * 1e-3)

    historical_state[1] = np.array(historical_state[1]) / 1000
    historical_state[7] = np.array(historical_state[7]) / 1000
    data = {'state': historical_state,
            'state_name': list_name
            }
    with open(folder + name + ".pkl", "wb") as data_handle:
        pickle.dump(data, data_handle)
        data_handle.close()
    hist_list = [historical_state]

    plot_state_solution(hist_list, list_name, folder, name, aux={8: energy_target}, plot_flag=plot_flag)
    plot_normal_tangent_velocity(hist_list, folder, name, plot_flag=plot_flag)
    # plot_orbit_solution(hist_list, ["Orbit"], a, b, rp, folder, name, plot_flag=plot_flag)
    # plot_general_solution(hist_list, ["General"], a, b, rp, folder, name, plot_flag=plot_flag)
    plt.show()