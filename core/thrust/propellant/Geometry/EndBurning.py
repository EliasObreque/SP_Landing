"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:40 PM 
els.obrq@gmail.com

"""
import numpy as np


class EndBurning(object):
    def __init__(self, di, diameter_ext, large, *args):
        self.current_burn_area = np.pi * (diameter_ext * 0.5) ** 2  # mm^2
        self.volume = self.current_burn_area * large  # mm^3
        self.diameter_ext = diameter_ext
        self.wall_web = large
        return

    def propagate_area(self, reg):
        self.get_transversal_area_at_reg(reg)

    def get_current_burn_area(self):
        return self.current_burn_area

    def get_transversal_area_at_reg(self, reg):
        outer = np.pi * (self.diameter_ext * 0.5) ** 2
        return outer

    def get_volume_at_reg(self, reg):
        return 0

    def get_core_perimeter_at_reg(self, reg):
        return 0

