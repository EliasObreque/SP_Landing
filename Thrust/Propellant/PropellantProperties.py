"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
TUBULAR = 'tubular'
BATES = 'bates'
STAR = 'star'
CUSTOM = 'custom'

# inhibit: 0:Top, 1: Bot, 2: Both

bates_geom = {'ext_diameter': 0.1,
              'int_diameter': 0.02,
              'large': 0.2,
              'Inhibit': 2}

star_geom = {'ext_diameter': 0.1,
             'int_diameter': 0.02,
             'n_points': 5,
             'star_angle_deg': 10,
             'large': 0.2,
             'Inhibit': 2}

tubular_geom = {'ext_diameter': 0.1,
                'int_diameter': None,
                'large': 0.2,
                'Inhibit': 0}

custom_geom = {'ext_diameter': 0.1,
               'int_diameter': 0.02,
               'large': 0.2,
               'Inhibit': 2}


propellant_properties = {'propellant_name': 'TRX-H609',
                         'geometry': bates_geom,
                         'propellant_geometry': BATES,
                         'isp_noise_std': None,
                         'isp_bias_std': None,
                         'isp_dead_time_max': None}