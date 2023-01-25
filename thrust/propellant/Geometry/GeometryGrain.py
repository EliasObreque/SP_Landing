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

end_burning = 'tubular'
bates = 'bates'
star = 'star'
CUSTOM = 'custom'


class GeometryGrain(object):
    def __init__(self, selected_geometry, grain_properties, *aux_dimension):
        self.selected_geometry = None
        self.diameter_int = grain_properties['int_diameter']
        self.diameter_ext = grain_properties['ext_diameter']
        self.large = grain_properties['large']
        self.inhibit = grain_properties['inhibit']
        self.current_reg_web = 0.0

        if selected_geometry == end_burning:
            self.selected_geometry = EndBurning(self.diameter_ext, self.large)
        elif selected_geometry == bates:
            self.selected_geometry = BATES(self.diameter_int, self.diameter_ext, self.large)
        elif selected_geometry == star:
            self.selected_geometry = STAR(self.diameter_int, self.diameter_ext, self.large, *aux_dimension)
        elif selected_geometry == CUSTOM:
            pass
        else:
            print('No geometry selected')
        return

    def get_grain_volume(self):
        return self.selected_geometry.volume

    def get_burning_area(self):
        return self.selected_geometry.get_current_burn_area()

    def calc_burn_area(self, reg):
        self.selected_geometry.propagate_area(reg)
        return




