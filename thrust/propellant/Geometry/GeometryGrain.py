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
    def __init__(self, selected_geometry, diameter_int, diameter_ext, large, propellant_properties, *aux_dimension):
        self.selected_geometry = None
        self.diameter_int = propellant_properties['geometry']['diameter_int']
        self.diameter_ext = propellant_properties['geometry']['diameter_ext']
        self.large = propellant_properties['geometry']['large']

        if selected_geometry == end_burning:
            self.selected_geometry = EndBurning(diameter_ext, large)
        elif selected_geometry == bates:
            self.selected_geometry = BATES(diameter_int, diameter_ext, large)
        elif selected_geometry == star:
            self.selected_geometry = STAR(diameter_int, diameter_ext, large, *aux_dimension)
        return




