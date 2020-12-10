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
                   'KNSU': {'density': 1.88, 'Isp': 164, 'burn_rate_constant': 8.26, 'pressure_exponent': 0.32,
                            'small_gamma': 1.133}}
TUBULAR = 'tubular'
BATES   = 'bates'
STAR    = 'star'


class PropellantGrain(object):
    def __init__(self, selected_propellant, diameter_int, diameter_ext, large, geometry_grain, *aux_dimension):
        self.selected_propellant = propellant_data[selected_propellant]
        if geometry_grain == TUBULAR:
            self.selected_geometry = self.calc_area_end_burn
        elif geometry_grain == BATES:
            self.selected_geometry = self.calc_area_bates
        elif geometry_grain == STAR:
            self.selected_geometry = self.calc_area_star
            self.n_point = aux_dimension[0]
            if len(aux_dimension) != 2:
                self.theta_star = self.calc_neutral_theta()
            else:
                self.theta_star = aux_dimension[1]
        self.diameter_ext = diameter_ext
        self.diameter_int = diameter_int
        self.large = large
        self.small_gamma = self.selected_propellant['small_gamma']
        self.big_gamma = self.calc_big_gamma()
        self.area_throat = 2
        self.init_area = 2
        self.init_mass = 2
        self.R_g = 250
        return

    def calc_mass_flow_propellant_(self, area_p, r_rate):
        den_p = self.selected_propellant['density']
        return den_p * area_p * r_rate

    def calc_max_pressure(self, area_throat, area_propellant):
        kn = area_propellant/area_throat
        c_char = self.selected_propellant['c_char']
        p_max = (kn * self.selected_propellant['burn_rate_constant'] * self.selected_propellant['density']
                 * c_char) ** (1/(1 - self.selected_propellant['pressure_exponent']))
        return p_max

    def simulate_profile(self, init_pressure, init_r, dt):
        init_state = [self.init_mass, init_pressure, init_r, self.init_area]
        mass = []
        pressure = []
        r_progress = []
        burn_area = []
        burning = True
        while burning:
            next_state = self.rungeonestep(self.thermal_model, init_state, dt)
            r_progress.append(next_state[0])
            mass.append(next_state[1])
            pressure.append(next_state[2])
            burn_area.append(next_state[3])
            if self.selected_propellant == TUBULAR:
                if next_state[0] >= self.large:
                    burning = False
            elif self.selected_propellant == BATES:
                if next_state[0] >= self.large:
                    burning = False
        return

    def thermal_model(self, state):
        r = state[0]
        m = state[1]
        p = state[2]
        area = state[3]
        density_p = self.selected_propellant['density']
        density_g = 0.001
        t = 1
        rhs = np.zeros(4)
        rhs[0] = self.selected_propellant['burn_rate_constant'] * p ** self.selected_propellant['pressure_exponent']
        rhs[1] = density_p * area * rhs[0]
        rhs[2] = area * rhs[0] * (density_p - density_g) - self.area_throat * p * self.big_gamma / np.sqrt(self.R_g * t)
        rhs[3] = self.selected_geometry(rhs[0])
        return rhs

    def calc_big_gamma(self):
        return np.sqrt(self.small_gamma * (2 / (self.small_gamma + 1)) ** ((self.small_gamma + 1)/
                                                                           (self.small_gamma - 1)))

    @staticmethod
    def calc_area_end_burn(*aux):
        return 0

    def calc_area_star(self, *aux):
        return (np.pi * 0.5 + np.pi / self.n_point - self.theta_star * 0.5) - 1 / np.tan(self.theta_star * 0.5) \
               * 2 * self.n_point * self.large * aux[0]

    def calc_area_bates(self, *aux):
        return 2 * np.pi * self.large * aux

    def calc_neutral_theta(self):
        def f(theta):
            return (np.pi * 0.5 + np.pi / self.n_point - theta * 0.5) - 1 / np.tan(theta * 0.5)
        return self.bisection(f, 0.001, np.pi)

    @staticmethod
    def rungeonestep(f, state, dt):
        x = np.array(state)
        k1 = f(x)
        xk2 = x + (dt / 2.0) * k1
        k2 = f(xk2)
        xk3 = x + (dt / 2.0) * k2
        k3 = f(xk3)
        xk4 = x + dt * k3
        k4 = f(xk4)
        next_x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_x

    @staticmethod
    def bisection(f, a, b, tol=1.0e-6):
        if a > b:
            raise ValueError("Intervalo mal definido")
        if f(a) * f(b) >= 0.0:
            raise ValueError("La función debe cambiar de signo en el intervalo")
        if tol <= 0:
            raise ValueError("La cota de error debe ser un número positivo")
        x = (a + b) / 2.0
        while True:
            if b - a < tol:
                return x
            # Utilizamos la función signo para evitar errores de precisión
            elif np.sign(f(a)) * np.sign(f(x)) > 0:
                a = x
            else:
                b = x
            x = (a + b) / 2.0


