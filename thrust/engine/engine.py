"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np


class Engine(object):
    amb_pressure = 1e-12

    def __init__(self, dt, thruster_properties):
        self.step = dt
        self.throat_diameter = thruster_properties['throat_diameter']
        self.diameter_ext = thruster_properties['case_diameter']
        self.case_large = thruster_properties['case_large']
        self.divergent_angle = np.deg2rad(thruster_properties['divergent_angle_deg'])
        self.convergent_angle = np.deg2rad(thruster_properties['convergent_angle_deg'])
        self.exit_nozzle_diameter = thruster_properties['exit_nozzle_diameter']

        d = 0.5 * (self.exit_nozzle_diameter - self.throat_diameter) / np.tan(self.convergent_angle)
        volume_convergent_zone = (np.pi * d * (self.diameter_ext * 0.5) ** 2) / 3
        volume_case = np.pi * ((self.diameter_ext * 0.5) ** 2) * self.case_large
        self.volume = volume_case + volume_convergent_zone
        self.chamber_temperature = 0.0
        self.chamber_pressure = 0.0
        self.exit_pressure = self.amb_pressure
        self.c_f = 0.0

        self.area_exit = np.pi * self.exit_nozzle_diameter ** 2 / 4
        self.area_th = np.pi * self.throat_diameter ** 2 / 4

    def get_chamber_pressure(self):
        return self.chamber_pressure

    def calc_thrust(self, burn_area):
        self.calc_chamber_pressure(burn_area)
        return self.c_f * self.chamber_pressure * self.area_th

    def calc_chamber_pressure(self, burn_area):
        self.chamber_pressure

    def calc_c_f(self, gamma):
        a = (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
        gamma_upper = np.sqrt(a * gamma)
        b = 2 * gamma ** 2 / (gamma - 1)
        ratio_p = self.exit_pressure / self.chamber_pressure
        c = (1 - ratio_p ** ((gamma - 1) / gamma))
        ratio_a = self.area_exit / self.area_th * (self.exit_pressure - self.amb_pressure) / self.chamber_pressure
        cf = np.sqrt(b * a * c) + ratio_a
        self.c_f = cf

    def calcKN(self):
        """Returns the motor's Kn when it has each grain has regressed by its value in regDepth, which should be a list
        with the same number of elements as there are grains in the motor."""
        x = np.linspace(0, self.grain.wallWeb, len(self.grain.dist_reg))
        burningSurfaceArea = self.grain.get_area_from_reg(x)
        self.engine_sim_res.channels['kn'] = burningSurfaceArea / self.throat_area
        return self.engine_sim_res.channels['kn']

    def check_geometric_cond(self):
        v1 = self.grain.volume / self.chamber_volume
        port_to_throat = self.grain.get_area_from_reg(0) / self.throat_area
        print("Port-to-Throat: ", port_to_throat, "- (propellant/Throat) = ", self.grain.get_area_from_reg(0), "/", self.throat_area)
        if port_to_throat > 2:
            print("ok")
        else:
            print("Port-to-Throat lower than 2")
        return port_to_throat
