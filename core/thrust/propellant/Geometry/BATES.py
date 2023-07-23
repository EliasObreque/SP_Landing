"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:40 PM 
els.obrq@gmail.com

"""
import numpy as np


class BATES(object):
    def __init__(self, core_diameter, diameter_ext, large, *args):
        self.large = large
        self.diameter_ext = diameter_ext
        self.diameter_int = core_diameter
        self.current_core_perimeter = 2 * np.pi * (core_diameter * 0.5)
        self.current_core_area = self.current_core_perimeter * large
        self.current_transversal_area = (np.pi * ((diameter_ext * 0.5) ** 2 - (core_diameter * 0.5) ** 2))
        self.volume = self.get_transversal_area_at_reg(0) * large
        self.wall_web = 0.5 * (self.diameter_ext - self.diameter_int)

    def propagate_area(self, reg):
        self.get_transversal_area_at_reg(reg)

    def get_transversal_area_at_reg(self, reg):
        outer = np.pi * (self.diameter_ext * 0.5) ** 2
        inner = np.pi * (self.diameter_int * 0.5 + reg) ** 2
        return outer - inner

    def get_core_perimeter_at_reg(self, reg):
        self.current_core_perimeter = 2 * np.pi * (self.diameter_int * 0.5 + reg)
        return max(0, self.current_core_perimeter)

    def get_current_burn_area(self):
        return 0.0