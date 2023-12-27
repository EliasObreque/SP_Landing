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
        return bool(np.linalg.norm(self.dynamic_model.current_pos_i) - self.r_moon < 2000.0)

    def notMass(self):
        return self.dynamic_model.current_mass < 5.0
