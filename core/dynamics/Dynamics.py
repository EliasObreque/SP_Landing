"""
Created: 9/15/2020
Author: Elias Obreque
email: els.obrq@gmail.com

"""
from .HamilCalcLimit import HamilCalcLimit
import numpy as np

from .OneDCoordinate import LinearCoordinate
from .PlaneCoordinate import PlaneCoordinate
from core.thrust.thruster import Thruster

ONE_D = '1D'
PLANE_D = '2D'
THREE_D = '3D'


class Dynamics(object):
    mu = 4.9048695e12  # m3s-2
    r_moon = 1.738e6
    g_planet = -1.607
    ge = 9.807

    def __init__(self, dt, mass, inertia, state, reference_frame, controller='basic_hamilton'):
        self.mass = mass
        self.controller_type = controller
        # self.step_width = dt
        # self.current_time = 0
        self.reference_frame = reference_frame
        if reference_frame == ONE_D:
            self.dynamic_model = LinearCoordinate(dt, self.g_planet, mass)
        elif reference_frame == PLANE_D:
            self.dynamic_model = PlaneCoordinate(dt, self.mu, self.r_moon, mass, inertia, state)
        else:
            print('Reference frame not selected')
        self.basic_hamilton_calc = HamilCalcLimit(mass, self.g_planet)
        self.thrusters = []
        self.controller_parameters = []
        self.controller_function = None

    def set_engines_properties(self, thruster_properties, propellant_properties, n_thrusters):
        self.thrusters = []
        for i in range(len(n_thrusters)):
            self.thrusters.append(Thruster(self.step_width, thruster_properties, propellant_properties))
        return

    def modify_individual_engine(self, n_engine, side, value):
        if side == 'alpha':
            self.thrusters[n_engine].set_alpha(value)
            self.basic_hamilton_calc.alpha = value
            self.basic_hamilton_calc.calc_parameters()
        elif side == 't_burn':
            self.thrusters[n_engine].set_t_burn(value)
        return

    # def run_simulation(self, x0, xf, time_options):
    #     x_states = [np.array(x0)]
    #     time_series = [time_options[0]]
    #     self.step_width = time_options[2]
    #     self.dynamic_model.dt = self.step_width
    #
    #     for thruster in self.thrusters:
    #         thruster.step_width = self.step_width
    #     thr = []
    #     m_dot = []
    #     index_control = []
    #     end_index_control = []
    #     hist_beta = []
    #     end_condition = False
    #     k = 0
    #     current_x = x0
    #     touch_surface = False
    #     land_index = 0
    #     while end_condition is False:
    #         total_thrust = 0
    #         m_dot_total = 0
    #         if self.controller_type == 'basic_hamilton':
    #             control_signal = self.basic_hamilton_calc.get_signal_control(current_x)
    #             self.thrusters[0].set_ignition(1 if np.sign(control_signal) < 0 else 0)
    #             self.thrusters[0].propagate_thrust()
    #             total_thrust = self.thrusters[0].get_current_thrust()
    #         elif self.controller_type == 'affine_function':
    #             # Get total thrust
    #             for j in range(len(self.thrusters)):
    #                 control_signal, beta = self.controller_function(self.controller_parameters[j], current_x,
    #                                                                 type_control='affine')
    #                 hist_beta.append(beta)
    #                 if control_signal == 1 and self.thrusters[j].current_beta == 0:
    #                     index_control.append(k)
    #                 if self.thrusters[j].thr_is_burned and self.thrusters[j].thr_is_on:
    #                     end_index_control.append(k - 1)
    #                 self.thrusters[j].set_ignition(control_signal)
    #                 self.thrusters[j].propagate_thrust()
    #                 total_thrust += self.thrusters[j].get_current_thrust()
    #                 m_dot_total += self.thrusters[j].get_current_m_flow()
    #         thr.append(total_thrust)
    #
    #         # dynamics
    #         next_x = self.dynamic_model.rungeonestep(current_x, total_thrust, m_dot_total)
    #
    #         # ....................
    #         k += 1
    #         current_x = next_x
    #         all_thrust_burned = [self.thrusters[j].thr_is_burned for j in range(len(self.thrusters))]
    #         if next_x[0] <= xf[0] and touch_surface is False:
    #             if np.abs(next_x[0]) <= np.abs(x_states[-1][0]):
    #                 land_index = k
    #             else:
    #                 land_index = k - 1
    #             touch_surface = True
    #         # else:
    #         if next_x[2] < 0:
    #             # True when the propellant uses all mass of the module. Error.
    #             end_condition = True
    #         elif (time_options[1] < k * self.step_width or (next_x[0] <= xf[0])) and np.all(all_thrust_burned):
    #             end_condition = True
    #             for h in range(len(end_index_control), len(index_control)):
    #                 end_index_control.append(k - 1)
    #             if land_index == len(x_states):
    #                 x_states.append(next_x)
    #                 time_series.append(time_options[0] + k * self.step_width)
    #                 thr.append(0)
    #         else:
    #             x_states.append(copy.copy(next_x))
    #             time_series.append(time_options[0] + k * self.step_width)
    #     if land_index == len(x_states):
    #         land_index -= 1
    #     if land_index == 0:
    #         land_index = -1
    #     return np.array(x_states), np.array(time_series), np.array(
    #         thr), index_control, end_index_control, land_index, hist_beta

    def calc_limits_by_single_hamiltonian(self, t_burn_min, t_burn_max, alpha_min, alpha_max, plot_data=False):
        self.basic_hamilton_calc.calc_limits_with_const_time(t_burn_min, alpha_min, alpha_max)
        self.basic_hamilton_calc.calc_limits_with_const_alpha(t_burn_min, t_burn_max, alpha_min)
        if plot_data:
            self.basic_hamilton_calc.show_time_limits(t_burn_min, t_burn_max)
            self.basic_hamilton_calc.show_alpha_limits(alpha_min, alpha_max)
        return

    def set_controller_parameters(self, parameters):
        self.controller_parameters = []
        for i in range(len(parameters[0])):
            self.controller_parameters.append([])
            for k in range(len(parameters)):
                self.controller_parameters[i].append(parameters[k][i])

    def get_current_state(self):
        return self.dynamic_model.get_current_state()

    def get_n_last_state(self, n):
        return self.dynamic_model.historical_pos_i[:-n]

    def isTouchdown(self) -> bool:
        return bool(np.linalg.norm(self.dynamic_model.current_pos_i) - 1000.0 - self.r_moon < 0.0)

    def notMass(self):
        return self.dynamic_model.current_mass < 5.0
