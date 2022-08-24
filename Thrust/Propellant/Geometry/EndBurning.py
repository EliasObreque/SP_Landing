"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:40 PM 
els.obrq@gmail.com

"""
import numpy as np


class EndBurning(object):
    def __init__(self, diameter_ext, large):
        self.init_area = np.pi * (diameter_ext * 0.5) ** 2  # mm^2
        self.volume_propellant = self.init_area * large  # mm^3
        self.free_volume = 0
        return

    @staticmethod
    def propagate_area():
        return 0

