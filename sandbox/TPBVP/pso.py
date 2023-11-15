"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
import multiprocessing
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
ra = 68e6
rp = 2e6
a = 0.5 * (ra + rp)
ecc = 1 - rp / a
b = a * np.sqrt(1 - ecc ** 2)
rm = 1.738e6
MAX_CORE = multiprocessing.cpu_count()
NCORE = int(MAX_CORE * 0.8)
matplotlib.rcParams.update({'font.size': 12})


class PSO:
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(1.5, 0.05, 0.2, .5), name=None):
        self.fitness_function = func
        self.dim = None
        self.name = name
        self.position = np.zeros(0)
        self.historical_position = []
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
        self.historical_g_position = []
        self.gbest_correlation = 0.0
        self.gbest_r2_score = 0.0
        self.gbest_mse = 0.0
        self.range_var = None
        self.evol_best_fitness = np.zeros(self.max_iteration)
        self.evol_p_fitness = np.zeros((self.npar, self.max_iteration))
        self.gbest_position = np.array([np.inf for _ in range(self.npar)])
        self.historical_fitness = []

    def initialize(self, range_var):
        self.range_var = range_var
        self.position = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])

        self.velocity = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])
        self.pbest_position = self.position

    def plot_state_solution(self, min_state_full, list_name, folder):
        for i, min_state in enumerate(np.array(min_state_full).T[1:]):
            fig = plt.figure()
            plt.grid()
            if list_name is not None:
                plt.ylabel(list_name[i])
            plt.xlabel("Time [s]")
            plt.plot(np.array(min_state_full).T[0], min_state)
            fig.savefig(folder + self.name + "_" + list_name[i].split(" ")[0] + '.pdf', format='pdf')

    def plot_historical_position(self, folder):
        fig, axes = plt.subplots(len(self.range_var), 1, sharex=True, figsize=(10, 5))
        plt.xlabel("Iteration")
        if len(self.range_var) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.set_ylabel("Particle {}".format(i + 1))
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.plot(np.arange(1, self.max_iteration + 1),
                    np.array(self.historical_position).T[i].T, '-.', lw=0.5, color='b')
            ax.plot(np.arange(1, self.max_iteration + 1),
                              np.array(self.historical_g_position).T[i], lw=1.0, color='r')
            ax.grid()
        fig.savefig(folder + self.name + "_hist_pos.pdf", format='pdf')

    def plot_best_cost(self, folder):
        fig = plt.figure()
        plt.plot(np.arange(1, len(self.evol_best_fitness) + 1), self.evol_best_fitness, 'red', lw=1)
        plt.plot(np.arange(1, len(self.evol_best_fitness) + 1), self.evol_p_fitness.T, '-.', color='blue', lw=0.5)
        plt.grid()
        plt.yscale("log")
        plt.ylabel("Evaluation cost")
        plt.xlabel("Iteration")
        fig.savefig(folder + self.name + "_hist_cost.pdf", format='pdf')

    @staticmethod
    def create_random_vector(range_var: list, vel=False) -> list:
        var = []
        for elem in range_var:
            temp = np.random.uniform(elem[0], elem[1])
            var.append(temp)
        return var

    def show_map2(self):
        xi, yi, zi = self.get_gridmap_approximation()
        # Crear una figura 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar la superficie
        surf = ax.plot_surface(xi, yi, zi, cmap='jet')

        # Agregar puntos originales
        ax.scatter(np.array(self.historical_position).T[0].flatten(),
                   np.array(self.historical_position).T[1].flatten(),
                   np.log10(self.historical_fitness).flatten(),
                   color='red', s=15, label='Positions')
        plt.colorbar(surf, ax=ax, label='fitness (Log10)')
        # Personalizar la apariencia
        ax.set_xlabel('Angular ignition [rad]')
        ax.set_ylabel('Engine Diameter [m]')
        ax.set_zlabel('fitness')
        ax.set_title('PSO evaluation')
        ax.legend()

    def show_map(self):
        cmap = plt.get_cmap('jet').reversed()  # I only want 4 colors from this cmap
        x = np.array(self.historical_position).T[0]
        y = np.array(self.historical_position).T[1]
        value = np.log10(self.historical_fitness)
        plt.figure()
        plt.scatter(x.T, y.T, c=value, cmap=cmap)
        plt.colorbar(label='Log10')

    def get_gridmap_approximation(self, res=100):
        x = np.array(self.historical_position).T[0].flatten()
        y = np.array(self.historical_position).T[1].flatten()
        value = np.log10(self.historical_fitness).flatten()
        xi = np.linspace(min(x), max(x), res)
        yi = np.linspace(min(y), max(y), res)
        xi, yi = np.meshgrid(xi, yi)
        # Interpolación de los puntos para obtener valores z en la superficie
        zi = griddata((x, y), value, (xi, yi), method='cubic', fill_value=np.max(value))
        return xi, yi, zi

    def plot_result(self, min_state, list_name, folder=""):
        self.plot_state_solution(min_state, list_name, folder)
        self.plot_historical_position(folder)
        self.plot_best_cost(folder)


class PSOStandard(PSO):
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(0.8, 0.01, 1.2, 1.5), name=None):
        super().__init__(func, n_particles, n_steps, parameters, name=name)

    def optimize(self, clip=True):
        iteration = 0
        W = self.w1
        min_state = None
        while iteration < self.max_iteration:
            self.historical_position.append(self.position.copy())
            pool = multiprocessing.Pool(processes=NCORE)
            result = pool.map(self.fitness_function, self.position)
            pool.close()
            # result = [self.fitness_function(pos) for pos in self.position]
            fitness = np.array([elem[0] for elem in result])
            self.historical_fitness.append(fitness)
            result = [elem[1] for elem in result]
            self.pbest_position[fitness < self.pbest_fitness_value] = self.position[fitness < self.pbest_fitness_value]
            self.pbest_fitness_value[fitness < self.pbest_fitness_value] = fitness[fitness < self.pbest_fitness_value]
            best_particle_idx = np.argmin(fitness)
            best_fitness = fitness[best_particle_idx]

            # print("BEST: ", best_fitness, self.position[best_particle_idx])
            if best_fitness < self.gbest_fitness_value:
                self.gbest_fitness_value = best_fitness
                self.gbest_position = self.position[best_particle_idx]
                min_state = result[best_particle_idx]

            self.historical_g_position.append(self.gbest_position)
            gbest = np.tile(self.gbest_position, (self.npar, 1))
            r = np.random.uniform(size=2)
            cognitive_comp = self.c1 * r[0] * (self.pbest_position - self.position)
            social_comp = self.c2 * r[1] * (gbest - self.position)
            self.velocity = W * self.velocity + cognitive_comp + social_comp
            self.position = self.velocity + self.position
            if clip:
                self.position = np.clip(self.position, np.array(self.range_var)[:, 0], np.array(self.range_var)[:, 1])

            W = self.w1 - (self.w1 - self.w2) * (iteration + 1) / self.max_iteration
            self.evol_best_fitness[iteration] = self.gbest_fitness_value
            self.evol_p_fitness[:, iteration] = self.pbest_fitness_value
            print("Train: ", iteration, "Fitness: ", self.gbest_fitness_value, "Worst: ", max(self.pbest_fitness_value), "Best:", self.gbest_position)
            iteration += 1
        print("Finished")
        return self.historical_g_position[-1], min_state

    def get_gains(self):
        return self.gbest_position


class APSO(PSOStandard):
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(1., 0.01, 1.2, 1.5)):
        super().__init__(func, n_particles, n_steps, parameters)
        self.c1_max = 1.2
        self.c1_min = 0.8
        self.c2_max = 1.5
        self.c2_min = 1.0

    def optimize(self, clip=True, list_name=None):
        iteration = 0
        W = self.w1
        c1 = self.c1_max
        c2 = self.c2_max
        modified_comp = 0
        min_state = None
        while iteration < self.max_iteration:
            self.historical_position.append(self.position.copy())
            pool = multiprocessing.Pool(processes=NCORE)
            result = pool.map(self.fitness_function, self.position)
            pool.close()
            # result = [self.fitness_function(pos) for pos in self.position]
            fitness = np.array([elem[0] for elem in result])
            self.historical_fitness.append(fitness)
            result = [elem[1] for elem in result]
            self.pbest_position[fitness < self.pbest_fitness_value] = self.position[fitness < self.pbest_fitness_value]
            self.pbest_fitness_value[fitness < self.pbest_fitness_value] = fitness[fitness < self.pbest_fitness_value]
            best_particle_idx = np.argmin(fitness)
            best_fitness = fitness[best_particle_idx]

            # print("BEST: ", best_fitness, self.position[best_particle_idx])
            if best_fitness < self.gbest_fitness_value:
                self.gbest_fitness_value = best_fitness
                self.gbest_position = self.position[best_particle_idx]
                min_state = result[best_particle_idx]
            self.historical_g_position.append(self.gbest_position)

            gbest = np.tile(self.gbest_position, (self.npar, 1))
            r = np.random.uniform(size=(self.npar, 2))
            cognitive_comp = c1 * np.diag(r[:, 0]) @ (self.pbest_position - self.position)
            social_comp = c2 * np.diag(r[:, 1]) @ (gbest - self.position)
            # modified_comp = W * c1 / c2 * (gbest - self.pbest_position)
            # my = np.array([g * pos for g, pos in zip(fitness, (gbest - self.position))])
            self.velocity = W * self.velocity + cognitive_comp + social_comp + modified_comp
            self.position = self.velocity + self.position
            self.position[:, 0] = self.position[:, 0] % (2 * np.pi)
            self.position[:, 1] = np.clip(self.position[:, 1], np.array(self.range_var)[1, 0],
                                          np.array(self.range_var)[1, 1])
            W = self.w1 - (self.w1 - self.w2) * (iteration + 1) / self.max_iteration
            c1 = self.c1_max - (self.c1_max - self.c1_min) * (iteration + 1) / self.max_iteration
            c2 = self.c2_max - (self.c2_max - self.c2_min) * (iteration + 1) / self.max_iteration

            self.evol_best_fitness[iteration] = self.gbest_fitness_value
            self.evol_p_fitness[:, iteration] = self.pbest_fitness_value
            print("Train: ", iteration, "Fitness: ", self.gbest_fitness_value, "Worst: ", max(self.pbest_fitness_value),
                  "Best:", self.gbest_position)
            iteration += 1
        print("Finished")

        self.plot_state_solution(min_state)
        self.plot_historical_position()
        return self.gbest_position

