"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:40 PM 
els.obrq@gmail.com

"""
import numpy as np


class BATES(object):
    def __init__(self, core_diameter, diameter_ext, large):
        self.large = large
        self.diameter_ext = diameter_ext
        self.diameter_int = core_diameter
        self.current_burn_area = 2 * np.pi * (core_diameter * 0.5) * large  # mm^2
        self.volume = (np.pi * ((diameter_ext * 0.5) ** 2 - (core_diameter * 0.5) ** 2)) * large  # mm^3
        return

    def propagate_area(self, reg):
        self.area_by_reg(reg)

    def get_current_burn_area(self):
        return self.current_burn_area

    def area_by_reg(self, reg):
        outer = np.pi * (self.diameter_ext * 0.5) ** 2
        inner = np.pi * (self.diameter_int * 0.5 + reg) ** 2
        return outer - inner
