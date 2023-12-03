"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:40 PM 
els.obrq@gmail.com

"""
import numpy as np


class RegTubular(object):
    def __init__(self, di, diameter_ext, large, *args):
        self.di = di
        self.area_i = np.pi * (di * 0.5) ** 2
        self.area_o = np.pi * (diameter_ext * 0.5) ** 2
        self.current_burn_area = np.pi * (diameter_ext * 0.5) ** 2  # mm^2
        self.volume = self.current_burn_area * large  # mm^3
        self.diameter_ext = diameter_ext
        self.wall_web = large

    def propagate_area(self, reg):
        self.get_transversal_area_at_reg(reg)

    def get_current_burn_area(self):
        return self.current_burn_area

    def get_transversal_area_at_reg(self, reg):
        return self.area_o - reg * (self.area_o - self.area_i)/self.wall_web

    def get_volume_at_reg(self, reg):
        current_burn_area = self.get_transversal_area_at_reg(reg)
        current_radio = (current_burn_area / np.pi) ** 0.5
        return (current_burn_area + self.area_i + self.di * current_radio) * (self.wall_web - reg) / 3

    def get_core_perimeter_at_reg(self, reg):
        return 0

