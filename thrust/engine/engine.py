"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
from thrust.propellant.propellant import Propellant
from scipy.optimize import fsolve


class Engine(object):
    amb_pressure = 1e-12

    def __init__(self, dt, thruster_properties, propellant_properties):
        self.step = dt
        self.t_ig = 0
        self.thr_is_on = False
        self.thr_is_burned = False
        self.current_time = 0
        self.current_burn_time = 0
        self.historical_mag_thrust = []
        self.current_mag_thrust_c = 0

        self.throat_diameter = thruster_properties['throat_diameter']
        self.diameter_ext = thruster_properties['case_diameter']
        self.case_large = thruster_properties['case_large']
        self.divergent_angle = np.deg2rad(thruster_properties['divergent_angle_deg'])
        self.convergent_angle = np.deg2rad(thruster_properties['convergent_angle_deg'])
        self.exit_nozzle_diameter = thruster_properties['exit_nozzle_diameter']

        d = 0.5 * (self.exit_nozzle_diameter - self.throat_diameter) / np.tan(self.convergent_angle)
        volume_convergent_zone = (np.pi * d * (self.diameter_ext * 0.5) ** 2) / 3
        volume_case = np.pi * ((self.diameter_ext * 0.5) ** 2) * self.case_large
        self.engine_volume = volume_case + volume_convergent_zone
        self.chamber_temperature = 0.0
        self.exit_pressure = self.amb_pressure
        self.chamber_pressure = self.exit_pressure
        self.c_f = 0.0

        self.area_exit = np.pi * self.exit_nozzle_diameter ** 2 / 4
        self.area_th = np.pi * self.throat_diameter ** 2 / 4
        self.propellant = Propellant(dt, propellant_properties)
        self.volume_free = self.engine_volume - self.propellant.get_grain_volume()
        self.init_stable_chamber_pressure = self.calc_chamber_pressure(self.propellant.get_burning_area())
        # TODO: fix check geometry
        # self.check_geometric_cond()

    def propagate_engine(self):
        p_c = self.get_chamber_pressure()
        # calc reg, and burn area
        self.propellant.propagate_grain(p_c)

        self.calc_chamber_pressure(self.propellant.get_burning_area())
        self.calc_exit_pressure(self.propellant.gamma)
        # calc CF
        self.calc_c_f(self.propellant.gamma)
        # calc thrust
        self.calc_thrust(self.propellant.get_burn_area())

    def get_chamber_pressure(self):
        return self.chamber_pressure

    def calc_thrust(self, burn_area):
        self.current_mag_thrust_c = self.c_f * self.chamber_pressure * self.area_th

    def calc_chamber_pressure(self, burn_area):
        area_ratio = burn_area / self.area_th
        pc = (self.propellant.burn_rate_constant * area_ratio * self.propellant.density *
              self.propellant.c_char) ** (1 / (1 - self.propellant.burn_rate_exponent))
        self.chamber_pressure = pc
        return pc

    def calc_exit_pressure(self, k, p_c=None):
        if p_c is None:
            p_c = self.get_chamber_pressure()
        """Solves for the nozzle's exit pressure, given an input pressure and the gas's specific heat ratio."""
        self.exit_pressure = fsolve(lambda x: (1 / self.calc_expansion()) - self.eRatioFromPRatio(k, x / p_c), 0)[0]
        return self.exit_pressure

    def calc_expansion(self):
        return (self.area_exit / self.area_th) ** 2

    @staticmethod
    def eRatioFromPRatio(k, pRatio):
        """Returns the expansion ratio of a nozzle given the pressure ratio it causes."""
        return (((k + 1) / 2) ** (1 / (k - 1))) * (pRatio ** (1 / k)) * (
                    (((k + 1) / (k - 1)) * (1 - (pRatio ** ((k - 1) / k)))) ** 0.5)

    def calc_c_f(self, gamma, p_c=None, exit_press=None):
        if p_c is None:
            p_c = self.chamber_pressure
        if exit_press is None:
            p_e = self.calc_exit_pressure(self.propellant.gamma, p_c)
        else:
            p_e = self.exit_pressure
        a = (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
        gamma_upper = np.sqrt(a * gamma)
        b = 2 * gamma ** 2 / (gamma - 1)
        ratio_p = p_e / p_c
        c = (1 - ratio_p ** ((gamma - 1) / gamma))
        ratio_a = self.area_exit / self.area_th * (p_e - self.amb_pressure) / p_c
        cf = np.sqrt(b * a * c) + ratio_a
        self.c_f = cf
        return self.c_f

    def get_c_f(self):
        return self.c_f

    def calc_kn(self, burning_surface_area):
        """Returns the motor's Kn when it has each grain has regressed by its value in regDepth, which should be a list
        with the same number of elements as there are grains in the motor."""
        return burning_surface_area / self.area_th

    def check_geometric_cond(self):
        init_burn_area = self.propellant.get_burning_area()
        port_to_throat = self.calc_kn(init_burn_area)
        print("Port-to-Throat: ", port_to_throat, "- (propellant/Throat) = ", self.grain.get_area_from_reg(0), "/", self.throat_area)
        if port_to_throat > 2:
            print("ok")
        else:
            print("Port-to-Throat lower than 2")
        return port_to_throat

    def reset_variables(self):
        self.t_ig = 0
        self.thr_is_on = False
        self.current_burn_time = 0
        self.current_time = 0
        self.current_mag_thrust_c = 0
        self.thr_is_burned = False
