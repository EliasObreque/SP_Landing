"""
Created: 9/15/2020
Author: Elias Obreque
email: els.obrq@gmail.com

"""
from .HamilCalcLimit import HamilCalcLimit
import numpy as np
from .LinearCoordinate import LinearCoordinate
from .PolarCoordinate import PolarCoordinate
from Thrust.Thruster import Thruster
ONE_D = '1D'
POLAR = 'polar'


class Dynamics(object):
    def __init__(self, dt, Isp, g_planet, mu_planet, r_planet, mass, reference_frame, controller='basic_hamilton'):
        self.mass = mass
        self.controller_type = controller
        self.step_width = dt
        self.current_time = 0
        self.Isp = Isp
        self.ge = 9.807
        self.mu = mu_planet
        self.r_moon = r_planet
        self.g_planet = g_planet
        self.c_char = Isp * self.ge
        self.reference_frame = reference_frame
        if reference_frame == ONE_D:
            self.dynamic_model = LinearCoordinate(dt, Isp, g_planet, mass)
        elif reference_frame == POLAR:
            self.dynamic_model = PolarCoordinate(dt, Isp, g_planet, mu_planet, r_planet, mass)
        else:
            print('Reference frame not selected')
        self.basic_hamilton_calc = HamilCalcLimit(self.mass, self.c_char, g_planet)
        self.thrusters = []
        self.controller_parameters = None
        self.controller_function = None

    def set_engines_properties(self, thruster_properties, propellant_properties):
        self.thrusters = []
        pulse_thruster = propellant_properties['pulse_thruster']
        for i in range(pulse_thruster):
            self.thrusters.append(Thruster(self.step_width, thruster_properties, propellant_properties))
            if thruster_properties['delay'] is not None:
                self.thrusters[i].lag_coef = thruster_properties['delay']
        return

    def modify_individual_engine(self, n_engine, side, value):
        if side == 'alpha':
            self.thrusters[n_engine].set_alpha(value)
            self.basic_hamilton_calc.alpha = value
            self.basic_hamilton_calc.calc_parameters()
        elif side == 't_burn':
            self.thrusters[n_engine].set_t_burn(value)
        return

    def run_simulation(self, x0, xf, time_options):
        x_states = [np.array(x0)]
        time_series = [time_options[0]]
        self.step_width = time_options[2]
        self.dynamic_model.dt = self.step_width

        for thruster in self.thrusters:
            thruster.step_width = self.step_width
        thr = []
        index_control = []
        end_index_control = []
        end_condition = False
        k = 0
        current_x = x0
        while end_condition is False:
            total_thrust = 0
            if self.controller_type == 'basic_hamilton':
                control_signal = self.basic_hamilton_calc.get_signal_control(current_x)
                self.thrusters[0].set_beta(1 if np.sign(control_signal) < 0 else 0)
                self.thrusters[0].propagate_thr()
                total_thrust = self.thrusters[0].get_current_thrust()
            elif self.controller_type == 'ga_wo_hamilton':
                # Get total thrust
                for j in range(len(self.thrusters)):
                    control_signal = self.controller_function(self.controller_parameters[j], current_x)
                    if control_signal == 1 and self.thrusters[j].current_beta == 0:
                        index_control.append(k)
                    if self.thrusters[j].thr_is_burned and self.thrusters[j].thr_is_on:
                        end_index_control.append(k - 1)
                    self.thrusters[j].set_beta(control_signal, n_engine=j)
                    self.thrusters[j].propagate_thr()
                    total_thrust += self.thrusters[j].get_current_thrust()
            thr.append(total_thrust)

            # dynamics
            next_x = self.dynamic_model.rungeonestep(current_x, total_thrust)

            # ....................
            k += 1
            current_x = next_x
            all_thrust_burned = [self.thrusters[j].thr_is_burned for j in range(len(self.thrusters))]
            if next_x[2] < 0:
                end_condition = True
            elif (time_options[1] < k * self.step_width or (next_x[0] <= xf[0])) and np.all(all_thrust_burned):
                end_condition = True
                for h in range(len(end_index_control), len(index_control)):
                    end_index_control.append(k - 1)
            else:
                x_states.append(next_x)
                time_series.append(time_options[0] + k * self.step_width)
        return np.array(x_states), np.array(time_series), np.array(thr), index_control, end_index_control

    def calc_limits_by_single_hamiltonian(self, t_burn_min, t_burn_max, alpha_min, alpha_max, plot_data=False):
        self.basic_hamilton_calc.calc_limits_with_const_time(t_burn_min, alpha_min, alpha_max)
        self.basic_hamilton_calc.calc_limits_with_const_alpha(t_burn_min, t_burn_max, alpha_min)
        if plot_data:
            self.basic_hamilton_calc.show_time_limits(t_burn_min, t_burn_max)
            self.basic_hamilton_calc.show_alpha_limits(alpha_min, alpha_max)
        return

    def set_controller_parameters(self, par_a, par_b):
        self.controller_parameters = [[par_a[i], par_b[i]] for i in range(len(par_b))]
        return
