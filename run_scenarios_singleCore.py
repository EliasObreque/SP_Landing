"""
Created: 9/14/2020
Author: Elias Obreque Sepulveda
email: els.obrq@gmail.com

array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)', 'TRX-H609', 'KNSU']

"""

from Scenarios.S1D_AFFINE.S1D_AFFINE import s1d_affine

NEUTRAL = 'neutral'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_ = 2000.0, 0.0, 100.0, 5.0, 30, 10

# Problem: "isp_noise"-"isp_bias"-"isp_bias-noise"-"state_noise"-"all" - "no_noise"
type_problem = "all"


def propagate(propellant_geometry):
    s1d_affine(propellant_geometry, type_problem, r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_)


# propagate(REGRESSIVE)
# propagate(NEUTRAL)
propagate(PROGRESSIVE)



