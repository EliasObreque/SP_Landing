"""
Created by:

@author: Elias Obreque
@Date: 12/7/2020 7:42 PM 
els.obrq@gmail.com

"""
from thrust.propellant.Geometry.GeometryGrain import GeometryGrain
from .source.propellant_data import propellant_data

import numpy as np
import sys

array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)',
                          'TRX-H609', 'KNSU']

TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
CUSTOM = 'custom'

square_mm2m = 1e-6
cube_mm2m = 1e-9
mm2m = 1e-3
g2kg = 1e-3
mm3_2_cm3 = 1e-3
gasConstant = 8314.0  # J/ kmol K
C2K = 273.15
ge = 9.807


class Propellant(GeometryGrain):
    def __init__(self, dt, propellant_properties, *aux_dimension):
        # propellant
        selected_propellant = propellant_properties['mixture_name']
        self.selected_propellant = [pro_data['data'] for pro_data in propellant_data if pro_data['name'] == selected_propellant][0]
        self.r_gases = gasConstant / self.selected_propellant['molecular_weight']
        self.gamma = self.selected_propellant['small_gamma']
        self.big_gamma = self.get_big_gamma()
        self.density = self.selected_propellant['density'] * 1e3    # kg/m3
        self.burn_rate_exponent = self.selected_propellant['pressure_exponent']
        self.burn_rate_constant = self.selected_propellant['burn_rate_constant'] * 1e-3  # m
        self.temperature = self.selected_propellant['temperature']  # K
        self.c_char = (1 / self.big_gamma) * (self.r_gases * self.temperature) ** 0.5   # m/s

        # Geometry
        self.dt = dt
        self.geometry_grain = propellant_properties['geometry']['type']
        if self.geometry_grain is not None:
            GeometryGrain.__init__(self, self.geometry_grain, propellant_properties['geometry']['setting'],
                                   *aux_dimension)
            # Mass properties:
            # mass
            self.mass = self.selected_geometry.volume * self.density
            # mass flux [kg/mm2]
            self.mass_flux = 0.0
            # mass flow [kg/s]
            self.mass_flow = 0.0
        else:
            print("Error defining grain geometry")

        # Noise
        self.isp0 = self.selected_propellant['Isp']
        self.std_noise = propellant_properties['isp_noise_std']
        self.std_bias = propellant_properties['isp_bias_std']

        self.v_exhaust = self.isp0 * ge     # average exhaust speed along the axis of the engine
        self.add_bias_isp()

    def propagate_grain(self, p_c: float):
        reg = self.dt * self.get_burn_rate(p_c)
        self.current_reg_web += reg
        # self.calc_burn_area(reg)

    def get_burn_area(self):
        return super().get_burning_area(self.current_reg_web)

    def get_burn_rate(self, p_c):
        return self.burn_rate_constant * p_c ** self.burn_rate_exponent + 0.0001

    def add_noise_isp(self):
        if self.std_noise is not None:
            noise_isp = np.random.normal(0, self.std_noise)
            return self.v_exhaust + noise_isp * ge
        else:
            return self.v_exhaust

    def add_bias_isp(self) -> None:
        if self.std_bias is not None:
            isp = np.random.normal(self.isp0, self.std_bias)
            self.v_exhaust = isp * ge

    def get_v_eq(self) -> float:
        return self.v_exhaust

    def calc_mass_flow_propellant(self, area_p, r_rate):
        den_p = self.density
        return den_p * area_p * r_rate

    def calc_max_pressure(self, area_throat, area_propellant):
        kn = area_propellant / area_throat
        kn = np.max(kn, 0)
        c_char = self.c_char
        p_max = (kn * self.burn_rate_constant * self.density
                 * c_char) ** (1 / (1 - self.selected_propellant['pressure_exponent']))
        return p_max

    def get_big_gamma(self):
        return np.sqrt(self.gamma * (2 / (self.gamma + 1)) ** ((self.gamma + 1) /
                                                               (self.gamma - 1)))

    def get_ise_temperature(self, t1, p1, p2):
        return t1 * (p1 / p2) ** ((1 - self.gamma) / self.gamma)

    def get_isp(self):
        return self.v_exhaust / ge

    def get_mass_at_reg(self, reg):
        return self.density * self.get_volume_at_reg(reg)

    def calculate_mass_properties(self, p_c):
        dreg = self.get_burn_rate(p_c) * self.dt
        self.mass_flux = self.get_mass_flux(self.current_reg_web, self.dt, self.mass_flow, dreg, self.density)
        new_mass = self.get_mass_at_reg(self.current_reg_web)
        self.mass_flow += (self.mass - new_mass) / self.dt
        self.mass = new_mass

    def reset_var(self):
        self.mass = self.get_mass_at_reg(0)
        self.mass_flow = 0.0
        self.mass_flux = 0.0
        self.current_reg_web = 0.0


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    propellant_grain = Propellant(array_propellant_names[2], 4, 30, 100, BATES, 3)
