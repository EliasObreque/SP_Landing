"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
from sklearn.metrics import r2_score
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
ra = 68e6
rp = 2e6
a = 0.5 * (ra + rp)
ecc = 1 - rp / a
b = a * np.sqrt(1 - ecc ** 2)
rm = 1.738e6


class PSORegression:
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(0.5, 0.05, 0.2, .5)):
        self.fitness_function = func
        self.dim = None
        self.position = []
        self.velocity = None
        self.max_iteration = n_steps
        # self.w = parameters[0]
        self.w1 = parameters[0]
        self.w2 = parameters[1]
        self.c1 = parameters[2]
        self.c2 = parameters[3]
        self.npar = n_particles
        self.pbest_position = self.position  # particle best position
        self.pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])
        self.gbest_fitness_value = float('inf')
        self.gbest_position = None  # gloal best position
        self.gbest_correlation = 0.0
        self.gbest_r2_score = 0.0
        self.gbest_mse = 0.0
        self.range_var = None
        self.corr_gain = 0.1
        self.mse_gain = 0.9
        self.entropy_gain = 0.0
        self.historical_loss = []
        self.historical_p = []
        return

    def initialize(self, range_var):
        self.range_var = range_var
        self.position = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])
        self.velocity = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])
        self.pbest_position = self.position
        self.gbest_position = np.array([np.inf for _ in range(self.npar)])
        self.evol_best_fitness = np.zeros(self.max_iteration)
        self.evol_p_fitness = np.zeros((self.npar, self.max_iteration))

    def remove_nans(self, var,  obj):
        ind_nan = ~np.isnan(obj)
        obj_res = obj[ind_nan]
        var_res = var[ind_nan]
        ind_nan = ~np.isnan(var_res)
        obj_res = obj_res[ind_nan]
        var_res = var_res[ind_nan]
        return  [var_res, obj_res]

    def get_historicals(self):
        return [self.historical_loss, self.historical_p]

    def cut_range(self, new_particle):
        for i, elem in enumerate(new_particle):
            if self.range_var[i][0] > elem:
                new_particle[i] = self.range_var[i][0]
            elif elem > self.range_var[i][1]:
                new_particle[i] = self.range_var[i][1]
        return new_particle

    def optimize(self, grav=False):
        iteration = 0
        W = self.w1
        gravity = 0
        while iteration < self.max_iteration:
            result = [self.fitness_function(pos) for pos in self.position]
            fitness = np.array([elem[0] for elem in result])
            result = [elem[1] for elem in result]
            self.pbest_position[fitness < self.pbest_fitness_value] = self.position[fitness < self.pbest_fitness_value]
            self.pbest_fitness_value[fitness < self.pbest_fitness_value] = fitness[fitness < self.pbest_fitness_value]
            best_particle_idx = np.argmin(fitness)
            best_fitness = fitness[best_particle_idx]
            # print("BEST: ", best_fitness, self.position[best_particle_idx])
            if best_fitness < self.gbest_fitness_value:
                self.gbest_fitness_value = best_fitness
                self.gbest_position = self.position[best_particle_idx]

            gbest = np.tile(self.gbest_position, (self.npar, 1))
            r = 0.01 * np.random.uniform(size=(self.npar, 2))
            cognitive_comp = self.c1 * np.diag(r[:, 0]) @ (self.pbest_position - self.position)
            social_comp = self.c2 * np.diag(r[:, 1]) @ (gbest - self.position)
            modified_comp = self.c1 / self.c2 * (gbest - self.pbest_position)
            mu = 0.01
            r = (gbest - self.position)
            r_norm = np.linalg.norm(r, axis=1)
            if grav:
                gravity = np.array([mu * r_/r_norm_ ** 3 if r_norm_ != 0 else r_ for r_, r_norm_ in zip(r, r_norm)])
                print("gravity: {}".format(np.sum(gravity, axis=0)))
            self.velocity = W * self.velocity + cognitive_comp + social_comp + modified_comp + gravity
            W = self.w1 - (self.w1 + self.w2) * iteration / self.max_iteration
            self.position = np.clip(self.velocity + self.position, np.array(self.range_var)[:, 0], np.array(self.range_var)[:, 1])
            self.evol_best_fitness[iteration] = self.gbest_fitness_value
            self.evol_p_fitness[:, iteration] = self.pbest_fitness_value
            print("Train: ", iteration, "Fitness: ", self.gbest_fitness_value, "Worst: ", max(self.pbest_fitness_value), "Best:", self.gbest_position)
            iteration += 1

            min_state = result[best_particle_idx]
            # fig_pso, ax_pso = plt.subplots(2, 2, figsize=(10, 8))
            # ax_pso = ax_pso.flatten()
            # ax_pso[0].plot(min_state[-1], [elem[0] for elem in min_state[0]])
            # ax_pso[1].plot(min_state[-1], [elem[1] for elem in min_state[0]])
            # ax_pso[2].plot(min_state[-1], min_state[2])
            # ax_pso[3].plot([elem[0] * 1e-3 for elem in min_state[0]], [elem[1] * 1e-3 for elem in min_state[0]])
            # ellipse = Ellipse(xy=(0, -(a - rp) * 1e-3), width=b * 2 * 1e-3, height=2 * a * 1e-3,
            #                   edgecolor='r', fc='None', lw=0.7)
            # ellipse_moon = Ellipse(xy=(0, 0), width=2 * rm * 1e-3, height=2 * rm * 1e-3, fill=True,
            #                        edgecolor='black', fc='gray', lw=0.4)
            # ax_pso[3].add_patch(ellipse)
            # ax_pso[3].add_patch(ellipse_moon)
            # plt.show()
        print("Finished")

        # Final selection
        eval_iter = [self.fitness_function(particle) for particle in self.position]

        # minimize
        fitness_value = eval_iter

        # if np.min(fitness_value) < self.gbest_fitness_value:
        #     index_min = np.argmin(fitness_value)
        #     self.gbest_fitness_value = fitness_value[index_min]
        #     self.gbest_position = self.position[index_min]
        #
        # final_eval = self.fitness_function(self.gbest_position)
        return self.gbest_position

    def get_gains(self):
        return self.gbest_position

    @staticmethod
    def create_random_vector(range_var: list, vel=False) -> list:
        var = []
        for elem in range_var:
            temp = np.random.uniform(elem[0], elem[1])
            var.append(temp)
        return var

    def set_particles_steps(self, n_par, n_steps):
        self.n_particles = n_par
        self.max_iteration = n_steps


def calc_rmse_norm(true_data, est_data):
    return np.sqrt(np.mean((true_data - est_data) ** 2))


def r2_score_function(x, y):
    temp = r2_score(x, y)
    return 1 - temp


def get_corr(x, y):
    return 1 - np.corrcoef(x, y)[0, 1]


def kernel_gauss(x, sigma):
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp((-(x*x)/(2*sigma*sigma)))


# Use Mercer Kernel on Maximum Correntropy for loss function
def correntropy(y_true, y_pred, sigma=1/np.sqrt(2*np.pi)):
    sum_score = kernel_gauss(y_true - y_pred, sigma)
    return np.mean(sum_score)