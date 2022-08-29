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


default_propellant = {'mixture_name': 'TRX-H609',
                      'geometry': {'type': BATES,
                                   'setting': bates_geom},
                      'isp_noise_std': None,
                      'isp_bias_std': None}


propellant_data = {'JPL_540A': {'density': 1.66, 'Isp': 280, 'burn_rate_constant': 5.13, 'pressure_exponent': 0.679,
                                'small_gamma': 1.2, 'molecular_weight': 25},
                   'ANP-2639AF': {'density': 1.66, 'Isp': 295, 'burn_rate_constant': 4.5, 'pressure_exponent': 0.313,
                                  'small_gamma': 1.18, 'molecular_weight': 24.7},
                   'CDT(80)': {'density': 1.74, 'Isp': 325, 'burn_rate_constant': 6.99, 'pressure_exponent': 0.48,
                               'small_gamma': 1.168, 'molecular_weight': 30.18},
                   'TRX-H609': {'density': 1.76, 'Isp': 300, 'burn_rate_constant': 4.92, 'pressure_exponent': 0.297,
                                'small_gamma': 1.21, 'molecular_weight': 25.97},
                   'KNSU': {'density': 1.88, 'Isp': 164, 'burn_rate_constant': 8.26, 'pressure_exponent': 0.32,
                            'small_gamma': 1.133, 'molecular_weight': 41.98}}