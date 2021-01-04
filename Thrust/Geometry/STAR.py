"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:40 PM 
els.obrq@gmail.com

"""
import numpy as np


class STAR(object):
    def __init__(self, diameter_int, diameter_ext, large, *aux_dimension):
        self.large = large
        self.diameter_ext = diameter_ext
        self.diameter_int = diameter_int
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
        self.volume_case = large * np.pi * (diameter_ext * 0.5) ** 2 - self.volume_propellant
        return

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

    def propagate_area(self, r_dot):
        return (np.pi * 0.5 + np.pi / self.n_point - self.theta_star * 0.5) - 1 / np.tan(self.theta_star * 0.5) \
               * 2 * self.n_point * self.large * r_dot

    def calc_neutral_theta(self):
        def f(theta):
            return (np.pi * 0.5 + np.pi / self.n_point - theta * 0.5) - 1 / np.tan(theta * 0.5)

        return self.bisection(f, 0.001, np.pi)
