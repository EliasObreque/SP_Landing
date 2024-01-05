"""
Created by Elias Obreque
Date: 05-12-2023
email: els.obrq@gmail.com
"""
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import pickle
from tools.pso import PSOStandard
# rmse
from sklearn.metrics import mean_squared_error
from core.module.Module import Module
from core.thrust.thruster import Thruster
from core.thrust.thrustProperties import second_thruster
from core.thrust.propellant.propellantProperties import second_propellant
from tools.Viewer import plot_state_solution, plot_normal_tangent_velocity, plot_pso_result

mass_0 = 24.0
rm = 1.738e6
inertia_0 = 1 / 12 * mass_0 * (0.2 ** 2 + 0.3 ** 2)
dt = 0.01
n_thruster = 2
t_end = 50
dataset = []
state_ = [np.array([0, rm + 5000]),
          np.array([0, 0]),
          0,
          0]
folder = "./logs/rw_tuning/"
name_ = "rw_pid"
# Thruster
second_thruster['max_ignition_dead_time'] = 0.5

# Propellant
second_propellant['isp_noise_std'] = 3.25
second_propellant['isp_bias_std'] = 10.83

thruster_pos = np.array([[-0.06975, -0.0887], [0.06975, -0.0887]]) + np.random.normal(0, 0.0000, size=2)

thruster_ang = np.random.normal(0, np.deg2rad(0.0), size=(len(thruster_pos)))
thruster_ang += np.array([-25 * np.pi / 180, 25 * np.pi / 180])

thruster_properties_ = [copy.deepcopy(second_thruster), copy.deepcopy(second_thruster)]
propellant_properties_ = [copy.deepcopy(second_propellant), copy.deepcopy(second_propellant)]

list_name = ["Position [m]", "Velocity [m/s]", "Mass [kg]", "Angle [deg]", "Angular velocity [deg/s]",
             "Inertia [kgm2]", "Thrust [N]", "Torque [mNm]", "Energy [kJ]", "beta [-]", "Angle Error [deg]",
             "Angular velocity Error [deg/s]", "RW velocity [deg/s]",
             "RW Torque [mNm]"]


def cost_model(gain_):
    gain_ = gain_[0]
    module = Module(mass_0, inertia_0, state_, 0, 0,
                    thruster_pos, thruster_ang, thruster_properties_,
                    propellant_properties_, "2D", dt, training=True)
    module.set_control_function([4980, 4980])
    module.control_pid.set_gain(gain_[0], gain_[1], gain_[2])
    historical_state_ = module.simulate(t_end, low_step=dt, progress=False, only_thrust=False, force_step=True,
                                       force_mode=1)
    # angular error in deg
    mech_cost = module.rw_model.historical_rw_velocity
    error = np.rad2deg(module.historical_theta_error)
    cost = mean_squared_error(np.zeros_like(error), error, squared=False) #+ np.mean(np.asarray(mech_cost) ** 2)

    historical_state_[3] = np.rad2deg(historical_state_[3])
    historical_state_[4] = np.rad2deg(historical_state_[4])
    historical_theta_error_ = np.rad2deg(module.historical_theta_error)
    historical_omega_error_ = np.rad2deg(module.historical_omega_error)
    historical_rw_velocity_ = np.rad2deg(module.rw_model.historical_rw_velocity)
    historical_state_.insert(-1, historical_theta_error_)
    historical_state_.insert(-1, historical_omega_error_)
    historical_state_.insert(-1, historical_rw_velocity_)
    return cost, historical_state_


if __name__ == '__main__':
    hist_list = []

    pso_rw = PSOStandard(cost_model, n_particles=30, n_steps=50)
    pso_rw.initialize([[-5, 5],
                       [-5, 5],
                       [-5, 5]])
    final_eval, best_state, hist_pos, hist_g_pos, eval_pos, eval_g_pos = pso_rw.optimize(clip=True, tol=1e-10)

    plot_pso_result(hist_pos, hist_g_pos, eval_pos, eval_g_pos, folder, name_ + "_train", plot_flag=False)
    plot_state_solution([best_state], list_name, folder, name_ + "_train", plot_flag=False)
    plot_normal_tangent_velocity([best_state], folder, name_ + "_train", plot_flag=False)

    for i in range(1):
        print(i)
        module = Module(mass_0, inertia_0, state_, 0, 0,
                        thruster_pos, thruster_ang, thruster_properties_,
                        propellant_properties_, "2D", dt, training=True)
        module.set_control_function([4950, 4950])
        module.control_pid.set_gain(pso_rw.gbest_position[0],  pso_rw.gbest_position[1], pso_rw.gbest_position[2])
        historical_state = module.simulate(t_end, low_step=dt, progress=False, only_thrust=False, force_step=True,
                                           force_mode=1)
        # angular error in deg
        historical_state[3] = np.rad2deg(historical_state[3])
        historical_state[4] = np.rad2deg(historical_state[4])
        historical_theta_error = np.rad2deg(module.historical_theta_error)
        historical_omega_error = np.rad2deg(module.historical_omega_error)
        historical_rw_velocity = np.rad2deg(module.rw_model.historical_rw_velocity)
        historical_state.insert(-1, historical_theta_error)
        historical_state.insert(-1, historical_omega_error)
        historical_state.insert(-1, historical_rw_velocity)
        torque_rw = np.array(module.rw_model.historical_rw_torque)
        ctrl_torque = np.array(module.rw_model.historical_rw_ctrl_torque)
        historical_state.insert(-1, np.vstack([torque_rw, ctrl_torque]).T * 1e3)
        hist_list.append(historical_state)

    plot_state_solution(hist_list, list_name, folder, name_, plot_flag=True)
    plot_normal_tangent_velocity(hist_list, folder, name_, plot_flag=True)
    plt.show()
