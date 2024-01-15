
"""
Created: 7/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
import numpy as np
from abc import abstractmethod, ABC
from scipy.optimize import fsolve

ge = 9.807


class BasicThruster(ABC):
    """
    thr_is_burned: State of burning process
    current_beta: Ignition control
    current_dead_time: Dead time from ignition control signal to the first increase in Thrust
    thr_is_on: Current state of the thrust
    current_time: Global current time
    t_ig: Ignition time @ global current time
    """

    def __init__(self, dt, thruster_properties, propellant_properties=None):
        """

        :param dt: Delta time of simulation
        :param thruster_properties: General and geometrical thrust properties
        :param propellant_properties: Grain geometry and mixture properties
        """
        self.thr_was_burned = False
        self.current_beta = 0
        self.current_dead_time = 0
        self.thr_is_on = False
        self.current_time = 0
        self.t_ig = 0
        self.thr_is_burning = False
        self.step_width = dt
        self.thrust_profile_type = thruster_properties['thrust_profile']['type']
        self.max_dead_time = thruster_properties['max_ignition_dead_time']
        self.ignition_dead_time = np.random.uniform(0, self.max_dead_time)
        if self.max_dead_time > 0 and self.ignition_dead_time < 1e-3:
            self.ignition_dead_time = 1e-3

    def reset_variables(self):
        """
        Reset all variables to create a new burning process
        """
        self.thr_was_burned = False
        self.current_beta = 0
        self.current_dead_time = 0
        self.thr_is_on = False
        self.thr_is_burning = False
        self.current_time = 0
        self.reset_dead_time()
        self.t_ig = 0

    def reset_dead_time(self):
        """
        Reset ignition dead time for a new burning process
        """
        self.ignition_dead_time = np.random.uniform(0, self.max_dead_time)
        self.current_dead_time = 0.0

    def set_dead_time(self, value):
        self.ignition_dead_time = value

    def set_thrust_on(self, value):
        """
        Change the state of thrust
        :param value: State of control ignition
        :return:
        """
        self.thr_is_on = value

    def set_ignition(self, beta):
        """
        Set the state of the thrust with the control signal
        :param beta:
        :return:
        """
        if self.thr_was_burned:
            self.current_beta = 0
            self.set_thrust_on(False)
        else:
            if beta == 1 and self.current_beta == 0:
                self.current_beta = beta
                self.thr_is_burning = 1

            if self.current_beta == 1:
                if self.current_dead_time >= self.ignition_dead_time:
                    self.t_ig = self.current_time
                    self.set_thrust_on(True)
                else:
                    self.update_dead_time()
            else:
                self.current_beta = 0
                self.thr_is_burning = 0
                self.set_thrust_on(False)

    def update_dead_time(self):
        """
        Update the counter of the dead time
        :return:
        """
        self.current_dead_time += self.step_width

    def calc_area_by_mass_flow(self, m_dot, isp, gamma, density, burn_rate_exponent, burn_rate_constant, c_char, throat_area):
        p_c_ = fsolve(lambda x: m_dot * isp * ge - self.calc_c_f(gamma, x, exit_press=None) * x * throat_area, 10000.0,
                      full_output=1)
        p_c = p_c_[0][0]
        area_p = throat_area * p_c ** (1 - burn_rate_exponent) / \
                 burn_rate_constant / density / c_char
        r = np.sqrt(area_p / np.pi)
        print("Pressure [kPa]:", p_c * 1e-3)
        print("Area: ", area_p, "Diameter:", r * 2)
        d = 1.5 * p_c * r / 110e6
        print(d)
        if d < 1e-3:
            d = 1e-3
        print("Thickness [mm]:", d * 1e3)
        print("Mass Engine: ", 2 * np.pi * d * r * 0.22 * 2700)
        return r * 1e3

    @abstractmethod
    def get_isp(self):
        """Get Isp mathematical model, File, or Engine"""

    @abstractmethod
    def propagate_thrust(self):
        """Update the thrust respect to a mathematical model, file or Engine"""

    @abstractmethod
    def get_current_thrust(self):
        """Get Thrust for mathematical model, File, or Engine"""

    @abstractmethod
    def get_current_m_flow(self):
        """
        Return mass flow [T/(Isp ge) (kg/s)]
        """