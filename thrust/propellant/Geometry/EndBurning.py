"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:40 PM 
els.obrq@gmail.com

"""
import numpy as np


class EndBurning(object):
    def __init__(self, diameter_ext, large):
        self.current_burn_area = np.pi * (diameter_ext * 0.5) ** 2  # mm^2
        self.volume = self.current_burn_area * large  # mm^3
        return

    def propagate_area(self, reg):
        self.area_by_reg(reg)

    def get_current_burn_area(self):
        return self.current_burn_area

    def area_by_reg(self, reg):
        pass

