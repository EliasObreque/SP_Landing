"""
Created by:

@author: Elias Obreque
@Date: 12/7/2020 7:42 PM 
els.obrq@gmail.com

"""
import numpy as np
array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)',
                          'TRX-H609', 'KNSU']
propellant_data = {'JPL_540A': {'density': 1.66, 'Isp': 280, 'burn_rate_constant': 5.13, 'pressure_exponent': 0.679,
                                'small_gamma': 1.2},
                   'ANP-2639AF': {'density': 1.66, 'Isp': 295, 'burn_rate_constant': 4.5, 'pressure_exponent': 0.313,
                                  'small_gamma': 1.18},
                   'CDT(80)': {'density': 1.74, 'Isp': 325, 'burn_rate_constant': 6.99, 'pressure_exponent': 0.48,
                               'small_gamma': 1.168},
                   'TRX-H609': {'density': 1.76, 'Isp': 300, 'burn_rate_constant': 4.92, 'pressure_exponent': 0.297,
                                'small_gamma': 1.21},
                   'KNSU': {'density': 1.88, 'Isp': 164, 'burn_rate_constant': 8.3, 'pressure_exponent': 0.32,
                            'small_gamma': 1.133}}


class PropellantGrain(object):
    def __init__(self, selected_propellant):
        self.selected_propellant = selected_propellant
        self.small_gamma = propellant_data[self.selected_propellant]['small_gamma']
        self.big_gamma = self.calc_big_gamma()
        return

    def calc_mass_flow_propellant_(self, area_p, r_rate):
        den_p = propellant_data[self.selected_propellant]['density']
        return den_p * area_p * r_rate

    def calc_steady_pressure(self):
        return

    def calc_big_gamma(self):
        return np.sqrt(self.small_gamma * (2 / (self.small_gamma + 1)) ** ((self.small_gamma + 1)/(self.small_gamma - 1)))

    def calc_area_end_burn(self, com_period_):
        return

    def calc_area_star(self, com_period_):
        return

    def calc_area_bates(self, com_period_):
        return

