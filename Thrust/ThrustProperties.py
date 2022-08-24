"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
engine_diameter_ext = 0.1
throat_diameter = 0.002  # m
e_nozzle_diameter = 0.005  # m
height = 0.2  # m
file_name = "Thrust/StarGrain7.csv"

default = {'throat_diameter': throat_diameter,
           'case_diameter': engine_diameter_ext,
           'case_large': height,
           'exit_nozzle_diameter': e_nozzle_diameter,
           'convergent_angle_deg': 30,
           'divergent_angle_deg': 15,
           'load_thrust_profile': None,
           'file_name': None,
           'dead_time': 0.2,
           'lag_coef': 0.5}
