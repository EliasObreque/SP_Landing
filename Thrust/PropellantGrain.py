"""
Created by:

@author: Elias Obreque
@Date: 12/7/2020 7:42 PM 
els.obrq@gmail.com

"""
from .Geometry.GeometryGrain import GeometryGrain
import numpy as np
import sys

array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)',
                          'TRX-H609', 'KNSU']
propellant_data = {'JPL_540A': {'density': 1.66, 'Isp': 280, 'burn_rate_constant': 5.13, 'pressure_exponent': 0.679,
                                'small_gamma': 1.2, 'molecular_weight': 25},
                   'ANP-2639AF': {'density': 1.66, 'Isp': 295, 'burn_rate_constant': 4.5, 'pressure_exponent': 0.313,
                                  'small_gamma': 1.18, 'molecular_weight': 24.7},
                   'CDT(80)': {'density': 1.74, 'Isp': 325, 'burn_rate_constant': 6.99, 'pressure_exponent': 0.48,
                               'small_gamma': 1.168, 'molecular_weight': 30.18},
                   'TRX-H609': {'density': 1.76, 'Isp': 300, 'burn_rate_constant': 4.92, 'pressure_exponent': 0.297,
                                'small_gamma': 1.21, 'molecular_weight': 25.97},
                   'KNSU': {'density': 1.88, 'Isp': 164, 'burn_rate_constant': 8.26, 'pressure_exponent': 0.32,
                            'small_gamma': 1.133, 'molecular_weight': 41.98}}
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
square_mm2m = 1e-6
cube_mm2m = 1e-9
mm2m = 1e-3
g2kg = 1e-3
mm3_2_cm3 = 1e-3
R_g = 8314.0  # J/kmol K
C2K = 273.15
ge  = 9.807


class PropellantGrain(GeometryGrain):
    def __init__(self, dt, propellant_properties, *aux_dimension):
        # Geometry
        self.geometry_grain = propellant_properties['geometry']
        if self.geometry_grain is not None:
            diameter_int = propellant_properties['geometry']['diameter_int']
            diameter_ext = propellant_properties['geometry']['diameter_ext']
            large        = propellant_properties['geometry']['large']
            GeometryGrain.__init__(self.geometry_grain, diameter_int, diameter_ext, large, *aux_dimension)
            volume_convergent       = (np.pi * 10 * (diameter_ext * 0.5) ** 2) / 3
            self.volume_case        = volume_convergent + self.selected_geometry.free_volume
            self.init_volume_case   = self.volume_case
            self.init_mass = self.density * self.selected_geometry.volume_propellant  # kg
        # Propellant
        selected_propellant = propellant_properties['propellant_name']
        self.selected_propellant = propellant_data[selected_propellant]
        self.isp0                = self.selected_propellant['Isp']
        self.std_sigma           = propellant_properties['isp_std']
        self.bias                = propellant_properties['isp_bias']
        self.c_char              = self.isp0 * ge
        self.r_gases             = R_g / self.selected_propellant['molecular_weight']
        self.small_gamma         = self.selected_propellant['small_gamma']
        self.big_gamma           = self.calc_big_gamma()
        self.density             = self.selected_propellant['density'] * 1e3
        self.area_throat = np.pi * 1 ** 2

    def set_throat_diameter(self, dia):
        self.area_throat = np.pi * (dia * 0.5) ** 2
        kn = self.selected_geometry.init_area / self.area_throat
        if kn < 200 or kn > 1000:
            print('Warning: kn out of range. Kn: ', kn)
            sys.exit()
        return

    def update_noise_isp(self):
        if self.std_sigma is not None:
            isp = np.random.normal(self.isp0, self.std_sigma)
            self.c_char = isp * ge

    def update_bias_isp(self):
        if self.bias is not None:
            isp = np.random.normal(self.isp0, self.bias)
            self.c_char = isp * ge

    def get_c_char(self):
        return self.c_char

    def calc_mass_flow_propellant_(self, area_p, r_rate):
        den_p = self.selected_propellant['density']
        return den_p * area_p * r_rate

    def calc_max_pressure(self, area_throat, area_propellant):
        kn = area_propellant / area_throat
        c_char = self.selected_propellant['c_char']
        p_max = (kn * self.selected_propellant['burn_rate_constant'] * self.selected_propellant['density']
                 * c_char) ** (1 / (1 - self.selected_propellant['pressure_exponent']))
        return p_max

    def simulate_profile(self, init_pressure, init_temperature, dt):
        init_state  = [0, self.init_mass, init_pressure, self.selected_geometry.init_area, self.init_volume_case]
        r_progress  = [0]
        mass        = [self.init_mass]
        pressure    = [init_pressure]
        burn_area   = [self.selected_geometry.init_area]
        volume      = [self.init_volume_case]
        temperature = [init_temperature + C2K]
        time        = [0]
        burning     = True
        last_state  = init_state
        last_temperature = init_temperature + C2K
        k = 0
        while burning:
            next_state = self.rungeonestep(self.propagate_grain_model, last_state, last_temperature, dt)
            r_progress.append(next_state[0])
            mass.append(next_state[1])
            pressure.append(next_state[2])
            burn_area.append(next_state[3])
            volume.append(next_state[4])
            new_temperature = self.get_ise_temperature(last_temperature, pressure[k], pressure[k + 1])
            temperature.append(new_temperature)
            last_state = next_state
            last_temperature = new_temperature
            k += 1
            time.append(dt * k)
            if self.selected_geometry_grain == TUBULAR:
                if next_state[0] >= self.large:
                    last_state[0] = 0
                    last_state[1] = 0
                    last_state[3] = 0
                    last_state[4] = self.volume_propellant + self.volume_case
            elif self.selected_geometry_grain == BATES:
                if next_state[0] + self.diameter_int * 0.5 >= self.diameter_ext * 0.5:
                    last_state[0] = 0
                    last_state[1] = 0
                    last_state[3] = 0
                    last_state[4] = self.volume_propellant + self.volume_case
            elif self.selected_geometry_grain == STAR:
                if next_state[0] + self.diameter_int * 0.5 >= self.diameter_ext * 0.5:
                    last_state[0] = 0
                    last_state[1] = 0
                    last_state[3] = 0
                    last_state[4] = self.volume_propellant + self.volume_case
            if next_state[2] <= pressure[0]:
                burning = False
        return [time, r_progress, mass, pressure, burn_area, volume, temperature]

    def propagate_grain_model(self, state, last_temperature):
        """
        :param state:
            r: [mmm]
            m: [kg]
            p: [Pa]
            area: [mm^2]
            vol: [mm^3]
            density: [g/cm^3]
        :param last_temperature:
            chamber temperature [K]
        :return:
        """
        r    = state[0]
        m    = state[1]
        p    = state[2]
        area = state[3]
        vol  = state[4]

        density_p = self.selected_propellant['density']
        density_g = 0.00001
        temp_c = last_temperature

        rhs = np.zeros(5)
        if area != 0:
            rhs[0] = self.selected_propellant['burn_rate_constant'] * (p * 1e-6) ** self.selected_propellant[
                'pressure_exponent']
        else:
            rhs[0] = 0
        rhs[1] = density_p * area * rhs[0] * mm3_2_cm3 * g2kg
        mass_diff = area * rhs[0] * (density_p - density_g) * mm3_2_cm3 * g2kg \
                    - self.area_throat * square_mm2m * p * self.big_gamma / np.sqrt(self.r_gases * temp_c)
        rhs[2] = mass_diff * self.r_gases * temp_c / (vol * cube_mm2m)
        if area != 0:
            rhs[3] = self.selected_geometry(rhs[0])
            rhs[4] = area * rhs[0]
        else:
            rhs[3] = 0
            rhs[4] = 0
        return rhs

    def calc_big_gamma(self):
        return np.sqrt(self.small_gamma * (2 / (self.small_gamma + 1)) ** ((self.small_gamma + 1) /
                                                                           (self.small_gamma - 1)))

    def rungeonestep(self, f, state, last_temperature, dt):
        x = np.array(state)
        k1 = f(x, last_temperature)
        xk2 = x + (dt / 2.0) * k1
        k2 = f(xk2, self.get_ise_temperature(last_temperature, state[2], xk2[2]))
        xk3 = x + (dt / 2.0) * k2
        k3 = f(xk3, self.get_ise_temperature(last_temperature, state[2], xk3[2]))
        xk4 = x + dt * k3
        k4 = f(xk4, self.get_ise_temperature(last_temperature, state[2], xk4[2]))
        next_x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_x

    def get_ise_temperature(self, t1, p1, p2):
        return t1 * (p1 / p2) ** ((1 - self.small_gamma) / self.small_gamma)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    propellant_grain = PropellantGrain(array_propellant_names[2], 4, 30, 100, BATES, 3)
    propellant_grain.set_throat_diameter(2)
    sim_data_sp = propellant_grain.simulate_profile(101325, 25, 0.001)
    """
    sim_data_sp:
    [time (0), r_progress (1), mass (2), pressure (3), burn_area (4), volume (5), temperature (6)]
    """
    plt.figure()
    plt.plot(sim_data_sp[0], sim_data_sp[1])
    plt.figure()
    plt.plot(sim_data_sp[0], sim_data_sp[2])
    plt.figure()
    plt.plot(sim_data_sp[0], sim_data_sp[3])
    plt.figure()
    plt.plot(sim_data_sp[0], sim_data_sp[4])
    plt.figure()
    plt.plot(sim_data_sp[0], sim_data_sp[5])
    plt.figure()
    plt.plot(sim_data_sp[0], sim_data_sp[6])
    plt.show()
