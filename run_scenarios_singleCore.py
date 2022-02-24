"""
Created: 9/14/2020
Author: Elias Obreque Sepulveda
email: els.obrq@gmail.com

array_propellant_names = ['JPL_540A', 'ANP-2639AF', 'CDT(80)', 'TRX-H609', 'KNSU']

"""
import sys
from Scenarios.S1D_AFFINE.S1D_AFFINE import s1d_affine

selected = 1

NEUTRAL = 'neutral'         # 3
PROGRESSIVE = 'progressive' # 2
REGRESSIVE = 'regressive'   # 1

r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_ = 2000.0, 0.0, 50.0, 5.0, 30, 10

# Problem: "isp_noise"-"isp_bias"-"isp_bias-noise"-"state_noise"-"all" - "no_noise"
type_problem = "all"


def propagate(propellant_geometry):
    s1d_affine(propellant_geometry, type_problem, r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_)


if len(sys.argv) != 1:
    selected = sys.argv[1]
else:
    selected = '1'

if selected == '1':
    propagate(REGRESSIVE)
elif selected == '2':
    propagate(PROGRESSIVE)
else:
    propagate(NEUTRAL)



