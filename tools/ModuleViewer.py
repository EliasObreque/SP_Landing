"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import Viewer as viewer


class ModuleViewer(object):
    def __init__(self, data_modules):
        self.modules = data_modules
        self.position = [module.dynamics.dynamic_model.historical_pos_i for module in self.modules]
        self.velocity = [module.dynamics.dynamic_model.historical_vel_i for module in self.modules]

    def plot_state(self):
        return
