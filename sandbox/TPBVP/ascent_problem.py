"""
Created by Elias Obreque
Date: 22-05-2023
email: els.obrq@gmail.com
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
import matplotlib.pyplot as plt

# Optimal Ascent Problem with MATLAB’s bvp4c
#
# by Jose J. Guzman and George E. Pollock
#
# This script uses MATLAB’s bvp4c to solve the problem of finding the
# optimal ascent trajectory for launch from a flat Moon to a 100 nautical
# mile circular orbit. In addition to this script, we use two functions:
# one to provide the differential equations and another that gives the
# boundary conditions:
#
# This file: OptimalAscent.m
# State and Costate Equations: ascent_odes_tf.m
# Boundary Conditions: ascent_bcs_tf.m
#

# Define parameters of the problem
h = 185.2e3  # meters, final altitude (100 nmi circular orbit)
Vc = 1.627e3  # m/s, Circular speed at 100 nmi
g_accel = 1.62  # m/sˆ2, gravitational acceleration of Moon
Thrust2Weight = 3  # Thrust to Weight ratio for Ascent Vehicle, in lunar G’s
rad2deg = 180/np.pi
#----------------------------------------------------------------------------
## Boundary Conditions
#----------------------------------------------------------------------------
# Initial conditions
# Launch from zero altitude with zero initial velocity
x0 = 0  # meters, initial x-position
y0 = 0  # meters, initial y-position
Vx0 = 0  # m/s, initial downrange velocity
Vy0 = 0  # m/s, initial vertical velocity
# Final conditions
yf = h  # meters, final altitude
Vxf = Vc  # m/s, final downrange velocity
Vyf = 0  # m/s, final vertical velocity
#----------------------------------------------------------------------------
## Initial Guesses
#----------------------------------------------------------------------------
# initial time
t0 = 0 
# list initial conditions in yinit, use zero if unknown
yinit = [x0, y0, Vx0, Vy0, 0, 0]  # guess for initial state and costate variables
tf_guess = 700  # sec, initial guess for final time
# Because the final time is free, we must parameterize the problem by
# the unknown final time, tf. Create a nondimensional time vector,
# tau, with Nt linearly spaced elements. (tau = time/tf) We will pass the
# unknown final time to bvp4c as an unknown parameter and the code will
# attempt to solve for the actual final time as it solves our TPBVP.
Nt = 100
tau = np.linspace(0, 1, Nt).T  # nondimensional time vector
# Create an initial guess of the solution using the MATLAB function
# bvpinit, which requires as inputs the (nondimensional) time vector,
# initial states (or guesses, as applicable), and the guess for the final
# time. The initial guess of the solution is stored in the structure
# solinit.

solinit = np.zeros((len(yinit), len(tau)))
#----------------------------------------------------------------------------
## Solution
#----------------------------------------------------------------------------
# Call bvp4c to solve the TPBVP. Point the solver to the functions
# containing the differential equations and the boundary conditions and
# provide it with the initial guess of the solution.

noise = 0



def ascent_odes_tf(tau, X, p):
    tf = p
    Acc = Thrust2Weight * g_accel + 5 * np.sin(tau * tf/100)
    xdot = X[2]
    ydot = X[3]
    Vxdot = (Acc + np.random.normal(0, noise)) * (1 / np.sqrt(1 + X[5] ** 2))
    Vydot = (Acc + np.random.normal(0, noise)) * (X[5] / np.sqrt(1 + X[5] ** 2)) - g_accel
    lambda2_bar_dot = 0 if xdot.size == 1 else np.zeros(len(xdot))
    lambda4_bar_dot = -X[4]
    dX_dtau = tf * np.array([xdot, ydot, Vxdot, Vydot, lambda2_bar_dot, lambda4_bar_dot])
    return dX_dtau


def ascent_bcs_tf(Y0, Yf, p):
    PSI = [Y0[1-1] - x0,    # Initial Condition
           Y0[2-1] - y0,    # Initial Condition
           Y0[3-1] - Vx0,    # Initial Condition
           Y0[4-1] - Vy0,    # Initial Condition
           Yf[2-1] - yf,    # Final Condition
           Yf[3-1] - Vxf,    # Final Condition
           Yf[4-1] - Vyf]    # Final Condition]
    return np.array(PSI)


def cost_function(particle):
    tf_ = particle[2]
    lambda_2 = particle[0]
    lambda_4 = particle[1]
    yinit = [x0, y0, Vx0, Vy0, lambda_2, lambda_4]
    temp_sol = solve_ivp(ascent_odes_tf, (0, 1), yinit, args=[tf_])
    error = ascent_bcs_tf(temp_sol.y[:, 0], temp_sol.y[:, -1], tf_)
    dy = ascent_odes_tf(temp_sol.t, temp_sol.y, [tf_])
    h = np.multiply(temp_sol.y[3, :], temp_sol.y[4, :])
    h += np.multiply(dy[0, :], np.ones(len(dy[0, :])))
    h += np.multiply(dy[1, :], temp_sol.y[5, :])
    return -np.mean(h)


sol = solve_bvp(ascent_odes_tf, ascent_bcs_tf, tau, solinit, p=[tf_guess])

# Extract the final time from the solution:
tf = sol.p[0]
noise = 1e-3
# Evaluate the solution at all times in the nondimensional time vector tau
# and store the state variables in the matrix Z.
Z = sol.sol(tau)[:6, :]
# Convert back to dimensional time for plotting
time = t0 + tau * (tf - t0)
# Extract the solution for each state variable from the matrix Z:
x_sol = Z[0, :]
y_sol = Z[1, :]
vx_sol = Z[2, :]
vy_sol = Z[3, :]

lambda2_bar_sol = Z[4, :]
lambda4_bar_sol = Z[5, :]
lamda_3 = 1

dvx = Thrust2Weight * g_accel * (1 / np.sqrt(1 + lambda4_bar_sol ** 2))
dvy = Thrust2Weight * g_accel * (lambda4_bar_sol / np.sqrt(1 + lambda4_bar_sol ** 2)) - g_accel
H = np.multiply(vy_sol, lambda2_bar_sol) + np.multiply(dvx, np.ones(len(vx_sol))) + np.multiply(dvy, lambda4_bar_sol)
print(tf, lambda2_bar_sol[0], lambda4_bar_sol[0])

fig, axes = plt.subplots(3, 2)
axes[0, 0].set_ylabel("X (m)")
axes[0, 0].plot(time, x_sol * 1e-3)
axes[0, 0].grid()
axes[0, 1].set_ylabel("Y (m)")
axes[0, 1].plot(time, y_sol * 1e-3)
axes[0, 1].grid()
axes[1, 0].set_ylabel("VX (m/s)")
axes[1, 0].plot(time, vx_sol * 1e-3)
axes[1, 0].grid()
axes[1, 1].set_ylabel("Vy (m/s)")
axes[1, 1].plot(time, vy_sol * 1e-3)
axes[1, 1].grid()
axes[2, 0].set_ylabel("Ang (m)")
axes[2, 0].plot(time, rad2deg * np.arctan(lambda4_bar_sol))
axes[2, 0].grid()
axes[2, 1].plot(time, lambda4_bar_sol)
axes[2, 1].grid()

plt.figure()
plt.plot(time, Thrust2Weight * g_accel + 5 * np.sin(time / 100))

plt.figure()
plt.plot(time, H)
plt.grid()
noise = 1e-3

#----------------------------------------------------------------------------
## Initial Guesses
#----------------------------------------------------------------------------
# initial time
t0 = 0
# list initial conditions in yinit, use zero if unknown
yinit = [x0, y0, Vx0, Vy0, 0, 0]  # guess for initial state and costate variables
tf_guess = 700  # sec, initial guess for final time
# Because the final time is free, we must parameterize the problem by
# the unknown final time, tf. Create a nondimensional time vector,
# tau, with Nt linearly spaced elements. (tau = time/tf) We will pass the
# unknown final time to bvp4c as an unknown parameter and the code will
# attempt to solve for the actual final time as it solves our TPBVP.
Nt = 100
tau = np.linspace(0, 1, Nt).T  # nondimensional time vector
# Create an initial guess of the solution using the MATLAB function
# bvpinit, which requires as inputs the (nondimensional) time vector,
# initial states (or guesses, as applicable), and the guess for the final
# time. The initial guess of the solution is stored in the structure
# solinit.

solinit = np.zeros((len(yinit), len(tau)))

sol = solve_bvp(ascent_odes_tf, ascent_bcs_tf, tau, solinit, p=[tf_guess])

# Extract the final time from the solution:
tf = sol.p[0]

# Evaluate the solution at all times in the nondimensional time vector tau
# and store the state variables in the matrix Z.
Z = sol.sol(tau)[:6, :]
# Convert back to dimensional time for plotting
time = t0 + tau * (tf - t0)
# Extract the solution for each state variable from the matrix Z:
x_sol = Z[0, :]
y_sol = Z[1, :]
vx_sol = Z[2, :]
vy_sol = Z[3, :]

lambda2_bar_sol = Z[4, :]
lambda4_bar_sol = Z[5, :]

lamda_3 = 1
H = np.multiply(y_sol, lambda2_bar_sol) + np.multiply(vx_sol, np.ones(len(vx_sol))) + np.multiply(vy_sol, lambda4_bar_sol)

print(tf, lambda2_bar_sol[0], lambda4_bar_sol[0])

fig, axes = plt.subplots(3, 2)
axes[0, 0].plot(time, x_sol * 1e-3)
axes[0, 0].grid()
axes[0, 1].plot(time, y_sol * 1e-3)
axes[0, 1].grid()
axes[1, 0].plot(time, vx_sol * 1e-3)
axes[1, 0].grid()
axes[1, 1].plot(time, vy_sol * 1e-3)
axes[1, 1].grid()
axes[2, 0].plot(time, rad2deg * np.arctan(lambda4_bar_sol))
axes[2, 0].grid()
axes[2, 1].plot(time, lambda4_bar_sol)
axes[2, 1].grid()
plt.figure()
plt.plot(time, H)
plt.grid()
#
# sol_pso = PSO(npar=200, nite=100, dim=3, ranges=[[-5.0, 5], [-5.0, 5], [300, 700]], func=cost_function)
# sol_pso.init_particles()
# sol_pso.run()
# best_solution = sol_pso.get_best_global()
# print(best_solution)
#
# tf_ = best_solution[-1]
# lambda_2, lambda_4 = best_solution[0], best_solution[1]
#
# yinit = [x0, y0, Vx0, Vy0, lambda_2, lambda_4]
# temp_sol = solve_ivp(ascent_odes_tf, (0, 1), yinit, args=[tf_], t_eval=np.linspace(0, 1, 100))
#
# time = temp_sol.t * tf_
# x_sol = temp_sol.y[0, :]
# y_sol = temp_sol.y[1, :]
# vx_sol = temp_sol.y[2, :]
# vy_sol = temp_sol.y[3, :]
#
# lambda2_bar_sol = temp_sol.y[4, :]
# lambda4_bar_sol = temp_sol.y[5, :]
#
# fig, axes = plt.subplots(3, 2)
# axes[0, 0].plot(time, x_sol * 1e-3)
# axes[0, 0].grid()
# axes[0, 1].plot(time, y_sol * 1e-3)
# axes[0, 1].grid()
# axes[1, 0].plot(time, vx_sol * 1e-3)
# axes[1, 0].grid()
# axes[1, 1].plot(time, vy_sol * 1e-3)
# axes[1, 1].grid()
# axes[2, 0].plot(time, rad2deg * np.arctan(lambda4_bar_sol))
# axes[2, 0].grid()
# axes[2, 1].plot(time, lambda4_bar_sol)
# axes[2, 1].grid()

plt.show()