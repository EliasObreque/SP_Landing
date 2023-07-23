"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 29-08-2022
"""


propellant_data = []


propellant_data.append({'name': 'JPL_540A',
                        'data': {'density': 1.66,
                                 'Isp': 280,
                                 'burn_rate_constant': 5.13,
                                 'pressure_exponent': 0.679,
                                 'small_gamma': 1.2,
                                 'molecular_weight': 25,
                                 'temperature': 1720.0,
                                 'minPressure': 0.0,
                                 'maxPressure': 10342500}})

propellant_data.append({'name': 'ANP-2639AF',
                        'data': {'density': 1.66,
                                 'Isp': 295,
                                 'burn_rate_constant': 4.5,
                                 'pressure_exponent': 0.313,
                                 'small_gamma': 1.18,
                                 'molecular_weight': 24.7,
                                 'temperature': 1720.0,
                                 'minPressure': 0.0,
                                 'maxPressure': 10342500}})

propellant_data.append({'name': 'CDT(80)',
                        'data': {'density': 1.74,
                                 'Isp': 325,
                                 'burn_rate_constant': 6.99,
                                 'pressure_exponent': 0.48,
                                 'small_gamma': 1.168,
                                 'molecular_weight': 30.18,
                                 'temperature': 1720.0,
                                 'minPressure': 0.0,
                                 'maxPressure': 10342500}})

propellant_data.append({'name': 'TRX-H609',
                        'data': {'density': 1.76,
                                 'Isp': 300,
                                 'burn_rate_constant': 4.92,
                                 'pressure_exponent': 0.297,
                                 'small_gamma': 1.21,
                                 'molecular_weight': 25.97,
                                 'temperature': 1720.0,
                                 'minPressure': 0.0,
                                 'maxPressure': 10342500}})

propellant_data.append({'name': 'Nakka - KNSU',
                        'data': {'density': 1.88,
                                 'Isp': 164,
                                 'burn_rate_constant': 0.101,  # mm/(s*Pa^n)
                                 'pressure_exponent': 0.319,
                                 'small_gamma': 1.133,
                                 'molecular_weight': 41.98,
                                 'temperature': 1720.0, # K
                                 'minPressure': 0.0,
                                 'maxPressure': 10342500}})
