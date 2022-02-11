"""
Created by:

@author: Elias Obreque
@Date: 8/11/2021 5:10 AM 
els.obrq@gmail.com

"""
import multiprocessing

from Scenarios.S1D_AFFINE.S1D_AFFINE import s1d_affine

NEUTRAL = 'neutral'
PROGRESSIVE = 'progressive'
REGRESSIVE = 'regressive'

r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_ = 2000.0, 0.0, 100.0, 5.0, 30, 10

# Problem: "isp_noise"-"isp_bias"-"isp_bias-noise"-"state_noise"-"all" - "no_noise"
type_problem = "all"


def propagate(propellant_geometry):
    s1d_affine(propellant_geometry, type_problem, r0_, v0_, std_alt_, std_vel_, n_case_train, n_thrusters_)


def main():
    p1 = multiprocessing.Process(target=propagate, args=[NEUTRAL])
    p2 = multiprocessing.Process(target=propagate, args=[PROGRESSIVE])
    p3 = multiprocessing.Process(target=propagate, args=[REGRESSIVE])

    p1.start()
    p2.start()
    p3.start()


if __name__ == '__main__':
    main()



