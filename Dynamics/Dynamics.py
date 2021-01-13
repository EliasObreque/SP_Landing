"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
from .HamilCalcLimit import HamilCalcLimit
import numpy as np
from tools.GeneticAlgorithm import GeneticAlgorithm
from .LinearCoordinate import LinearCoordinate
from .PolarCoordinate import PolarCoordinate
from Thrust.Thruster import Thruster
ONE_D = '1D'
POLAR = 'polar'


class Dynamics(object):
    def __init__(self, dt, Isp, g_planet, mu_planet, r_planet, mass, reference_frame):
        self.mass = mass
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

    def set_engine_properties(self, thruster_properties, propellant_properties):
        pulse_thruster = propellant_properties['pulse_thruster']
        for i in range(pulse_thruster):
            self.thrusters.append(Thruster(self.step_width, thruster_properties, propellant_properties))
        return

    def run_simulation(self, x0, xf, time_options):
        x_states = [np.array(x0)]
        time_series = [time_options[0]]
        thr = [0]
        end_condition = False
        k = 0
        current_x = x0
        while end_condition is False:
            total_thrust = 0
            if self.thrusters[0].selected_propellant.geometry_grain is None:
                control_signal = self.basic_hamilton_calc.get_signal_control(current_x)
                self.thrusters[0].set_beta(1 if np.sign(control_signal) < 0 else 0)
                self.thrusters[0].propagate_thr()
                total_thrust += self.thrusters[0].get_current_thrust()
            else:
                print('GA used')
                control_signal = 0
                # Get total thrust
                for thruster in self.thrusters:
                    thruster.set_beta(1 if np.sign(control_signal) < 0 else 0)
                    thruster.propagate_thr()
                    total_thrust += thruster.get_current_thrust()

            # dynamics
            next_x = self.dynamic_model.rungeonestep(current_x, total_thrust)
            # Solid rocket engine
            # ....................
            k += 1
            current_x = next_x
            if time_options[1] < k * self.step_width or (next_x[0] < xf[0] and thr[k - 1] == 0.0):
                end_condition = True
            else:
                x_states.append(next_x)
                time_series.append(time_options[0] + k * self.step_width)
                thr.append(total_thrust)
        return np.array(x_states), np.array(time_series), np.array(thr)

    def calc_optimal_parameters(self, init_state, max_generation, n_individuals, range_variables):
        ga = GeneticAlgorithm(0.1, self.g_planet, init_state, max_generation, n_individuals, range_variables)
        ga.optimize(self.dynamic_model.rungeonestep, self.c_char)
        return

    def calc_limits_by_single_hamiltonian(self, t_burn_min, t_burn_max, alpha_min, alpha_max, plot_data=False):
        self.basic_hamilton_calc.calc_limits_with_const_time(t_burn_min, alpha_min, alpha_max)
        self.basic_hamilton_calc.calc_limits_with_const_alpha(t_burn_min, t_burn_max, alpha_min)
        if plot_data:
            self.basic_hamilton_calc.show_time_limits(t_burn_min, t_burn_max)
            self.basic_hamilton_calc.show_alpha_limits(alpha_min, alpha_max)
        return

