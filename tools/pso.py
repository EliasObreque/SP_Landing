"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
from sklearn.metrics import r2_score
import random


class PSORegression:
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(2.5, 0.5, 0.8, 0.2)):
        self.func = func
        self.dim = None
        self.particle_position = None
        self.particle_velocity = None
        self.max_iteration = n_steps
        # self.w = parameters[0]
        self.w1 = parameters[0]
        self.w2 = parameters[1]
        self.c1 = parameters[2]
        self.c2 = parameters[3]
        self.n_particles = n_particles
        self.pbest_position = self.particle_position  # particle best position
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
        self.fitness_function = r2_score_function
        return

    def initialize(self, range_var):
        self.range_var = range_var
        self.particle_position = np.array(
            [self.create_random_vector(range_var) for _ in range(self.n_particles)])
        self.particle_velocity = np.array(
            [[0.0] * len(range_var) for _ in range(self.n_particles)])
        self.gbest_position = np.array([1.0 for _ in range(len(range_var))])

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

    def optimize(self):
        i_iter = 0
        stagnation_counter = 0
        w = self.w1
        while i_iter < self.max_iteration:
            if i_iter > 0.6 * self.max_iteration:
                w -= (self.w1 - self.w2) / self.max_iteration
            eval_iter = [self.func(particle) for particle in self.particle_position]

            fitness_value = eval_iter
            fit_position = []
            for i, fit_part in enumerate(fitness_value):
                if fit_part < self.pbest_fitness_value[i]:
                    fit_position.append([fit_part, self.particle_position[i]])
                else:
                    fit_position.append([self.pbest_fitness_value[i], self.particle_position[i]])

            self.pbest_fitness_value = np.array([elem[0] for elem in fit_position])
            self.historical_p.append(self.pbest_fitness_value)
            self.pbest_position = np.array([elem[1] for elem in fit_position])

            if np.min(fitness_value) < self.gbest_fitness_value:
                index_min = np.argmin(fitness_value)
                self.gbest_fitness_value = fitness_value[index_min]
                self.gbest_position = self.particle_position[index_min]
            else:
                stagnation_counter += 1
            self.historical_loss.append(self.gbest_fitness_value)
            # print(self.max_iteration * (n_e - 1) + i_iter, self.gbest_fitness_value)
            if stagnation_counter >= 10:
                self.particle_velocity += np.random.normal(0, 1.0, size=np.shape(self.particle_velocity))
                stagnation_counter = 0
            else:
                self.particle_velocity = w * self.particle_velocity + \
                                         (self.c1 * random.random()) * (self.pbest_position - self.particle_position) + \
                                         (self.c2 * random.random()) * (self.gbest_position - self.particle_position)
            new_position = self.particle_velocity + self.particle_position
            new_position = np.array([self.cut_range(new_) for new_ in new_position])
            self.particle_position = new_position
            # self.particle_position[np.where(np.abs(self.particle_position) < 1e-2)] = 0.1
            i_iter += 1
            print(i_iter, self.gbest_fitness_value, self.gbest_mse)

        # Final selection
        eval_iter = [self.func(particle) for particle in self.particle_position]

        # minimize
        fitness_value = eval_iter

        if np.min(fitness_value) < self.gbest_fitness_value:
            index_min = np.argmin(fitness_value)
            self.gbest_fitness_value = fitness_value[index_min]
            self.gbest_position = self.particle_position[index_min]

        final_eval = self.func(self.gbest_position)
        return final_eval

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