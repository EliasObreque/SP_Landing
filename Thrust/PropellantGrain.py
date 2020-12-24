"""
Created by:

@author: Elias Obreque
@Date: 12/7/2020 7:42 PM 
els.obrq@gmail.com

"""
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


class PropellantGrain(object):
    def __init__(self, selected_propellant, diameter_int, diameter_ext, large, geometry_grain, *aux_dimension):
        self.diameter_ext = diameter_ext
        self.diameter_int = diameter_int
        self.large = large
        self.selected_propellant = propellant_data[selected_propellant]
        self.r_gases = R_g / self.selected_propellant['molecular_weight']
        volume_convergent = (np.pi * 10 * (diameter_ext * 0.5) ** 2) / 3
        self.volume_case = volume_convergent
        self.selected_geometry_grain = geometry_grain
        if geometry_grain == TUBULAR:
            self.selected_geometry = self.calc_area_end_burn
            self.init_area = np.pi * (diameter_ext * 0.5) ** 2  # mm^2
            self.volume_propellant = self.init_area * large  # mm^3
            self.init_mass = self.selected_propellant['density'] * self.volume_propellant * mm3_2_cm3 * g2kg  # kg
            self.volume_case += 0  # mm^3
        elif geometry_grain == BATES:
            self.selected_geometry = self.calc_area_bates
            self.init_area = 2 * np.pi * (diameter_int * 0.5) * large  # mm^2
            self.volume_propellant = (np.pi * ((diameter_ext * 0.5) ** 2 - (diameter_int * 0.5) ** 2)) * large  # mm^3
            self.init_mass = self.selected_propellant['density'] * self.volume_propellant * mm3_2_cm3 * g2kg  # kg
            self.volume_case += large * np.pi * (diameter_ext * 0.5) ** 2 - self.volume_propellant
        elif geometry_grain == STAR:
            self.selected_geometry = self.calc_area_star
            self.n_point = aux_dimension[0]
            if len(aux_dimension) != 2:
                self.theta_star = self.calc_neutral_theta()
            else:
                self.theta_star = aux_dimension[1]
            seg = diameter_int * 0.5 * (np.sin(np.pi / self.n_point) / np.sin(self.theta_star / 2))
            self.init_area = 2 * self.n_point * large * seg
            area_triangle = self.calc_triangle_area(self.diameter_int)
            self.volume_propellant = large * np.pi * (diameter_ext * 0.5) ** 2 \
                                     - large * area_triangle * self.n_point * 2
            self.volume_case += large * np.pi * (diameter_ext * 0.5) ** 2 - self.volume_propellant

        self.init_volume_case = self.volume_case
        self.small_gamma = self.selected_propellant['small_gamma']
        self.big_gamma = self.calc_big_gamma()
        self.area_throat = np.pi * 1 ** 2
        kn = self.init_area / self.area_throat
        if kn < 200 or kn > 1000:
            print('Warning: kn out of range. Kn: ', kn)
            sys.exit()
        return

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
        init_state = [0, self.init_mass, init_pressure, self.init_area, self.init_volume_case]
        r_progress = [0]
        mass = [self.init_mass]
        pressure = [init_pressure]
        burn_area = [self.init_area]
        volume = [self.init_volume_case]
        temperature = [init_temperature + C2K]
        time = [0]
        burning = True
        last_state = init_state
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
                    last_state[3] = 0
            elif self.selected_geometry_grain == BATES:
                if next_state[0] + self.diameter_int * 0.5 >= self.diameter_ext * 0.5:
                    last_state[3] = 0
            elif self.selected_geometry_grain == STAR:
                if next_state[0] + self.diameter_int * 0.5 >= self.diameter_ext * 0.5:
                    last_state[3] = 0
            if next_state[2] <= init_state[2]:
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
        r = state[0]
        m = state[1]
        p = state[2]
        area = state[3]
        vol = state[4]

        density_p = self.selected_propellant['density']
        density_g = 0.00001
        temp_c = last_temperature

        rhs = np.zeros(5)
        rhs[0] = self.selected_propellant['burn_rate_constant'] * (p * 1e-6) ** self.selected_propellant[
            'pressure_exponent']
        rhs[1] = density_p * area * rhs[0] * mm3_2_cm3 * g2kg
        mass_diff = area * rhs[0] * (density_p - density_g) * mm3_2_cm3 * g2kg \
                    - self.area_throat * square_mm2m * p * self.big_gamma / np.sqrt(self.r_gases * temp_c)
        rhs[2] = mass_diff * self.r_gases * temp_c / (vol * cube_mm2m)
        rhs[3] = self.selected_geometry(rhs[0])
        rhs[4] = area * rhs[0]
        return rhs

    def calc_big_gamma(self):
        return np.sqrt(self.small_gamma * (2 / (self.small_gamma + 1)) ** ((self.small_gamma + 1) /
                                                                           (self.small_gamma - 1)))

    def calc_area_star(self, r_dot):
        return (np.pi * 0.5 + np.pi / self.n_point - self.theta_star * 0.5) - 1 / np.tan(self.theta_star * 0.5) \
               * 2 * self.n_point * self.large * r_dot

    def calc_area_bates(self, r_dot):
        return 2 * np.pi * self.large * r_dot

    def calc_neutral_theta(self):
        def f(theta):
            return (np.pi * 0.5 + np.pi / self.n_point - theta * 0.5) - 1 / np.tan(theta * 0.5)

        return self.bisection(f, 0.001, np.pi)

    def calc_triangle_area(self, current_diameter_int):
        seg = current_diameter_int * 0.5 * (np.sin(np.pi / self.n_point) / np.sin(self.theta_star / 2))
        temp = self.cos_theorem(seg, current_diameter_int * 0.5, self.theta_star * 0.5 - np.pi / self.n_point)
        s = (seg + current_diameter_int * 0.5 + temp) * 0.5
        h = 2 / (current_diameter_int * 0.5) * np.sqrt(s * (s - seg) * (s - current_diameter_int * 0.5) * (s - temp))
        area_triangle = current_diameter_int * 0.5 * h * 0.5
        return area_triangle

    @staticmethod
    def cos_theorem(a, b, gamma):
        c = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(gamma))
        return c

    @staticmethod
    def calc_area_end_burn(*aux):
        return 0

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

    @staticmethod
    def bisection(f, a, b, tol=1.0e-6):
        if a > b:
            raise ValueError("Poorly defined interval")
        if f(a) * f(b) >= 0.0:
            raise ValueError("The function must change sign in the interval")
        if tol <= 0:
            raise ValueError("The error bound must be a positive number")
        x = (a + b) / 2.0
        while True:
            if b - a < tol:
                return x
            elif np.sign(f(a)) * np.sign(f(x)) > 0:
                a = x
            else:
                b = x
            x = (a + b) / 2.0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    propellant_grain_endburn = PropellantGrain(array_propellant_names[2], 4, 30, 100, BATES, 8)
    sim_data_sp = propellant_grain_endburn.simulate_profile(101325, 25, 0.01)
    """
    sim_data_sp:
    [time, r_progress, mass, pressure, burn_area, volume, temperature]
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
