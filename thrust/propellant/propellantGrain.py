"""
Created by:

@author: Elias Obreque
@Date: 12/7/2020 7:42 PM 
els.obrq@gmail.com

"""
from thrust.propellant.Geometry.GeometryGrain import GeometryGrain
from .PropellantProperties import propellant_data

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
R_g = 8314.0  # J/kmol K
C2K = 273.15
ge = 9.807


class PropellantGrain(GeometryGrain):
    def __init__(self, dt, propellant_properties, *aux_dimension):
        # propellant
        selected_propellant = propellant_properties['propellant_name']
        self.selected_propellant = propellant_data[selected_propellant]
        self.r_gases = R_g / self.selected_propellant['molecular_weight']
        self.small_gamma = self.selected_propellant['small_gamma']
        self.big_gamma = self.get_big_gamma()
        self.density = self.selected_propellant['density'] * 1e3

        # Geometry
        self.dt = dt
        self.geometry_grain = propellant_properties['geometry']
        if self.geometry_grain is not None:
            GeometryGrain.__init__(self.geometry_grain, propellant_properties, *aux_dimension)
            self.mass = self.selected_geometry.volume * self.density

        # Noise
        self.isp0 = self.selected_propellant['Isp']
        self.std_noise = propellant_properties['isp_noise_std']
        self.std_bias = propellant_properties['isp_bias_std']

        self.v_exhaust = self.isp0 * ge     # average exhaust speed along the axis of the engine
        self.add_bias_isp()

    def add_noise_isp(self):
        if self.std_noise is not None:
            noise_isp = np.random.normal(0, self.std_noise)
            return self.v_exhaust + noise_isp * ge
        else:
            return self.v_exhaust

    def add_bias_isp(self):
        if self.std_bias is not None:
            isp = np.random.normal(self.isp0, self.std_bias)
            self.v_exhaust = isp * ge

    def get_v_eq(self):
        return self.v_exhaust

    def calc_mass_flow_propellant_(self, area_p, r_rate):
        den_p = self.selected_propellant['density']
        return den_p * area_p * r_rate

    def calc_max_pressure(self, area_throat, area_propellant):
        kn = area_propellant / area_throat
        c_char = self.selected_propellant['c_char']
        p_max = (kn * self.selected_propellant['burn_rate_constant'] * self.selected_propellant['density']
                 * c_char) ** (1 / (1 - self.selected_propellant['pressure_exponent']))
        return p_max

    def get_big_gamma(self):
        return np.sqrt(self.small_gamma * (2 / (self.small_gamma + 1)) ** ((self.small_gamma + 1) /
                                                                           (self.small_gamma - 1)))

    def get_ise_temperature(self, t1, p1, p2):
        return t1 * (p1 / p2) ** ((1 - self.small_gamma) / self.small_gamma)

    def get_isp(self):
        return self.v_exhaust / ge


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    propellant_grain = PropellantGrain(array_propellant_names[2], 4, 30, 100, BATES, 3)
