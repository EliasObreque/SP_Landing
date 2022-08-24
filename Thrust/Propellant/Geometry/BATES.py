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
        self.init_area = 2 * np.pi * (core_diameter * 0.5) * large  # mm^2
        self.volume = (np.pi * ((diameter_ext * 0.5) ** 2 - (core_diameter * 0.5) ** 2)) * large  # mm^3
        return

    def propagate_area(self, r_dot):
        return 2 * np.pi * self.large * r_dot
