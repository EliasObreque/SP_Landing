"""
Created by:

@author: Elias Obreque
@Date: 12/25/2020 4:38 PM 
els.obrq@gmail.com

"""
from abc import abstractmethod
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

    def isGrain(self):
        return True if self.selected_geometry else False

    def get_grain_volume(self):
        return self.selected_geometry.volume

    def get_burning_area(self, reg):
        transversal_area = self.selected_geometry.get_transversal_area_at_reg(reg)
        core_area = self.selected_geometry.get_core_perimeter_at_reg(reg) * self.get_length_at_reg(reg)

        exposed_face = 2
        if self.inhibit['top'] and self.inhibit['bot']:
            exposed_face = 0
        elif self.inhibit['top'] or self.inhibit['bot']:
            exposed_face = 1
        return core_area + transversal_area * exposed_face

    def calc_burn_area(self, reg):
        self.selected_geometry.propagate_area(reg)

    def get_volume_at_reg(self, reg):
        if ~isinstance(self.selected_geometry, EndBurning):
            area = self.selected_geometry.get_transversal_area_at_reg(reg)
            length = self.get_length_at_reg(reg)
            return area * length
        else:
            # TODO: add divergence angle of area
            area = self.selected_geometry.get_transversal_area_at_reg(reg)
            length = self.get_length_at_reg(reg)
            return area * length

    def get_length_at_reg(self, reg):
        large_pos = self.get_large_pos(reg)
        return large_pos[1] - large_pos[0]

    def get_large_pos(self, reg):
        # inhibit: True 1, False 0
        if self.inhibit['top'] and self.inhibit['bot']:
            return [0, self.large]
        elif self.inhibit['top']:
            return [0, self.large - reg]
        elif self.inhibit['bot']:
            return [reg, self.large]
        else:
            return [reg, self.large - reg]

    def get_web_left(self, reg):
        wall_left = self.selected_geometry.wall_web - reg
        length_left = self.get_length_at_reg(reg)
        return min(wall_left, length_left)

    def get_port_area(self, reg):
        face_area = self.selected_geometry.get_transversal_area_at_reg(reg)
        uncored_area = self.get_circle_area(self.selected_geometry.diameter_ext)
        return uncored_area - face_area

    def get_mass_flux(self, reg, dt, mass_flow, dreg, density):
        """Uses the grain's mass flux method to return the max. Assumes that it will be at the port of the grain!"""
        end_position = self.get_large_pos(reg)

        diameter = self.selected_geometry.diameter_ext
        # If a position above the top face is queried, the mass flow is just the input mass and the
        # diameter is the casting tube
        if end_position[1] < end_position[0]:
            return mass_flow / self.get_circle_area(diameter)
        # If a position in the grain is queried, the mass flow is the input mass, from the top face,
        # and from the tube up to the point. The diameter is the core.
        if end_position[1] <= end_position[1]:
            if sum(self.inhibit.values()) == 2:
                top = 0
                countedCoreLength = end_position[1]
            else:
                top = self.selected_geometry.get_transversal_area_at_reg(reg + dreg) * dreg * density
                countedCoreLength = end_position[1] - (end_position[0] + dreg)
            # This block gets the mass of propellant the core burns in the step.
            core = ((self.get_port_area(reg + dreg) * countedCoreLength)
                - (self.get_port_area(reg) * countedCoreLength))
            core *= density

            mass_flow += ((top + core) / dt)
            return mass_flow / self.get_port_area(reg + dreg)
        # A position past the grain end was specified, so the mass flow includes the input mass flow
        # and all mass produced by the grain. Diameter is the casting tube.
        mass_flow += (self.get_volume_slice(reg, dreg) * density / dt)
        return mass_flow / self.get_circle_area(diameter)

    def get_volume_slice(self, reg, dreg):
        return self.get_volume_at_reg(reg) - self.get_volume_at_reg(reg + dreg)


    @staticmethod
    def get_circle_area(diameter):
        return np.pi * (diameter * 0.5) ** 2



