"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:38 PM 
els.obrq@gmail.com

"""
from .BATES import BATES
from .STAR import STAR
from .EndBurning import EndBurning
import numpy as np
end_burning = 'end_burning'
bates = 'bates'
star = 'star'


class GeometryGrain(object):
    def __init__(self, selected_geometry, diameter_int, diameter_ext, large, *aux_dimension):
        self.selected_geometry = None
        if selected_geometry == end_burning:
            self.selected_geometry = EndBurning(diameter_ext, large)
        elif selected_geometry == bates:
            self.selected_geometry = BATES(diameter_int, diameter_ext, large)
        elif selected_geometry == star:
            self.selected_geometry = STAR(diameter_int, diameter_ext, large, *aux_dimension)
        return




