"""
Created: 7/15/2020
Author: Elias Obreque
email: els.obrq@gmail.com
"""

import numpy as np
from scipy.stats import rankdata
from matplotlib import pyplot as plt
from tools.MonteCarlo import MonteCarlo
from copy import deepcopy
plt.rcParams["font.family"] = "Times New Roman"
plt.ion()


class GeneticAlgorithm(object):
    def __init__(self, max_generation=10, n_individuals=10, ranges_variable=None, mutation_probability=0.1):
        self.Ah = 0.10
        self.Bh = 1.0
        self.Ch = 0.1
        self.selection_method = 'rank'
        self.current_cost = []
        self.max_generation = max_generation
        self.n_individuals = n_individuals
        self.range_variables = ranges_variable
        self.n_variables = len(ranges_variable)
        self.ind_selected = []
        self.weight_crossover = 0.3
        self.n_thrust = self.range_variables[0][3]
        self.mutation_probability = mutation_probability
        self.thrust_comp = None
        self.population = []
        self.historical_cost = []
        self.function_update = None
        self.init_state = None
        self.cost_function = None
        self.save_impact_velocity = None
        self.ga_dynamics = None
        self.time_options = None
        self.propellant_properties = None
        self.thruster_properties = None
        self.create_first_population()

    def create_first_population(self):
        for k in range(self.n_individuals):
            individual = []
            for i in range(self.n_variables):
                if self.range_variables[i][0] == 'float':
                    temp = np.random.uniform(self.range_variables[i][1], self.range_variables[i][2])
                    individual.append(temp)
                elif self.range_variables[i][0] == 'int':
                    if len(self.range_variables[i]) == 3:
                        temp = np.random.randint(self.range_variables[i][1], self.range_variables[i][2])
                    else:
                        temp = self.range_variables[i][1]
                        self.n_thrust = temp
                    individual.append(temp)
                elif self.range_variables[i][0] == 'str':
                    type_pro = np.random.randint(1, len(self.range_variables[i]))
                    temp = self.range_variables[i][type_pro]
                    individual.append(temp)
                elif self.range_variables[i][0] == 'float_iter':
                    temp = []
                    for m in range(int(self.range_variables[i][3])):
                        # temp.append(np.random.uniform(self.range_variables[i][1], self.range_variables[i][2]))
                        # loc = 0.5 * (self.range_variables[i][1] + self.range_variables[i][2])
                        mean = 0.5 * (self.range_variables[i][2] - self.range_variables[i][1])
                        if self.range_variables[i][1] < self.range_variables[i][2]:
                            left = self.range_variables[i][1]
                            right = self.range_variables[i][2]
                            mean = 0.5 * (self.range_variables[i][2] - self.range_variables[i][1])
                        else:
                            left = self.range_variables[i][2]
                            right = self.range_variables[i][1]
                            mean = 0.5 * (self.range_variables[i][1] - self.range_variables[i][2])
                        temp.append(np.random.triangular(left,
                                                         left + mean * 0.7,
                                                         right))
                    individual.append(temp)
            self.population.append(individual)
        return

    def optimize(self, cost_function=None, n_case=1, restriction_function=None, alt_noise=None):
        self.cost_function = cost_function
        self.ga_dynamics   = restriction_function[0]
        self.init_state    = restriction_function[1:3]
        self.time_options  = restriction_function[3]
        self.ga_dynamics.set_engines_properties(restriction_function[5], restriction_function[4], self.population[0][2])

        print('Running...')
        generation = 1
        states, time_data, Tf, index_control, end_index_control, land_index = self.ga_evaluate(self.population,
                                                                                               n_case, alt_noise)
        temp = np.mean(self.current_cost, 1) + np.std(self.current_cost, 1)
        print('Generation: ', generation, ', Cost: ', min(temp))
        self.historical_cost.append(self.current_cost[int(np.argmin(temp))])
        percent = 0.8
        n_indiv_by_selec = round(self.n_individuals * percent)
        rest_ind = n_indiv_by_selec % 2
        n_indiv_by_selec -= rest_ind

        while generation < self.max_generation:
            next_population = []
            crossover_arith = True
            if self.n_individuals - n_indiv_by_selec != 0:
                descent = self.ga_selection(self.n_individuals - n_indiv_by_selec, direct_method=True)
                next_population = next_population + descent
            for i in range(int(n_indiv_by_selec * 0.5)):
                index_parents = self.ga_selection(2)
                if crossover_arith:
                    descent1, descent2 = self.ga_crossover_arithmetic(index_parents[0], index_parents[1])
                    crossover_arith = False
                else:
                    descent1, descent2 = self.ga_crossover_coding(index_parents[0], index_parents[1])
                    crossover_arith = True

                new_generation1 = self.ga_mutation(descent1, True)
                new_generation2 = self.ga_mutation(descent2, True)
                next_population.append(new_generation1)
                next_population.append(new_generation2)
            del self.population
            self.population = deepcopy(next_population)
            states, time_data, Tf, index_control, end_index_control, land_index = self.ga_evaluate(next_population,
                                                                                                   n_case,
                                                                                                   alt_noise)
            generation += 1
            if generation > self.max_generation:
                self.max_generation = generation
            temp = np.mean(self.current_cost, 1) + np.std(self.current_cost, 1)
            print('Generation: ', generation, ', Cost: ', min(temp))
            self.historical_cost.append(self.current_cost[int(np.argmin(temp))])

        best_index = int(np.argmin(temp))
        best_states, best_Tf = states[best_index], Tf[best_index]
        best_individuals = self.population[best_index]
        best_index_control = index_control[best_index]
        best_end_index_control = end_index_control[best_index]
        best_time_data = time_data[best_index]
        best_landing_index = land_index[best_index]
        self.plot_cost(n_case)
        return best_states, best_time_data, best_Tf, best_individuals, best_index_control,\
               best_end_index_control, best_landing_index

    def plot_cost(self, n_case):
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Optimization function')
        for k in range(n_case):
            plt.plot(np.arange(1, self.max_generation + 1), np.array(self.historical_cost)[:, k], lw=0.8)
        plt.grid()

    def ga_evaluate(self, next_population, n_case, alt_noise_):
        self.current_cost = []
        X_states = []
        THR      = []
        IC       = []
        EC       = []
        TIME     = []
        LAND_INDEX = []
        self.ga_dynamics.controller_function = self.get_beta

        # # Generation of case (Monte Carlo)
        rN = []
        vN = []
        mN = []
        alt_noise = False
        if alt_noise_ is not None:
            alt_noise = alt_noise_[0]
            sdr = alt_noise_[1]
            sdv = 0
            sdm = 0
            rN = MonteCarlo(self.init_state[0][0], sdr, n_case).random_value()
            vN = MonteCarlo(self.init_state[0][1], sdv, n_case).random_value()
            mN = MonteCarlo(self.init_state[0][2], sdm, n_case).random_value()

        for indv in range(self.n_individuals):
            X_states.append([])
            THR.append([])
            IC.append([])
            EC.append([])
            TIME.append([])
            LAND_INDEX.append([])
            self.current_cost.append([])
            self.ga_dynamics.set_controller_parameters(next_population[indv][3], next_population[indv][4])

            for j in range(len(self.ga_dynamics.thrusters)):
                self.ga_dynamics.modify_individual_engine(j, 'alpha', next_population[indv][0][j])
                self.ga_dynamics.modify_individual_engine(j, 't_burn', next_population[indv][1][j])

            for k in range(n_case):
                if alt_noise:
                    x0 = [rN[k], vN[k], mN[k]]
                else:
                    x0 = self.init_state[0]

                x_states, time_series, thr, index_control, end_index_control, land_i = self.ga_dynamics.run_simulation(
                    x0,
                    self.init_state[1],
                    self.time_options)

                j_cost = self.cost_function(x_states, thr, land_i, self.Ah, self.Bh)
                self.current_cost[indv].append(j_cost)
                X_states[indv].append(x_states)
                LAND_INDEX[indv].append(land_i)
                THR[indv].append(thr)
                TIME[indv].append(time_series)
                IC[indv].append(index_control)
                EC[indv].append(end_index_control)
                # Reset thruster
                for thrust in self.ga_dynamics.thrusters:
                    thrust.reset_variables()
        return X_states, TIME, THR, IC, EC, LAND_INDEX

    @staticmethod
    def get_beta(control_par, current_state):
        a = control_par[0]
        b = control_par[1]
        current_alt = current_state[0]
        current_vel = current_state[1]
        f = a * current_alt + b * current_vel
        if f <= 0:
            return 1
        else:
            return 0

    def ga_selection(self, n, direct_method=False):
        if direct_method is False:
            temp = np.mean(self.current_cost, 1) + np.std(self.current_cost, 1)
            if self.selection_method == 'roulette':
                probability_selection = np.array(temp) / np.sum(temp)
                ind_selected = np.random.choice(a=np.arange(self.n_individuals),
                                                size=n,
                                                p=list(probability_selection),
                                                replace=True)
            elif self.selection_method == "rank":
                ranks = rankdata(1 * np.array(temp))
                selection_probability = 1 / ranks
                selection_probability = selection_probability / np.sum(selection_probability)
                ind_selected = np.random.choice(a=np.arange(self.n_individuals),
                                                size=n,
                                                p=list(selection_probability),
                                                replace=True)
            else:
                print('Select method')
                ind_selected = []
        else:
            temp = np.mean(self.current_cost, 1) + np.std(self.current_cost, 1)
            list_order = sorted(temp)
            value_select = list_order[:n]
            ind_selected = []
            for elem in value_select:
                index_select = np.where(temp == elem)[0][0]
                ind_selected.append(deepcopy(self.population[int(index_select)]))
        return ind_selected

    def ga_crossover_coding(self, parents_1, parents_2):
        children_1 = []
        children_2 = []
        cut_position = np.random.randint(0, self.n_variables)
        children_1[:cut_position] = deepcopy(self.population[parents_1][:cut_position])
        children_1[cut_position:] = deepcopy(self.population[parents_2][cut_position:])
        children_2[:cut_position] = deepcopy(self.population[parents_2][:cut_position])
        children_2[cut_position:] = deepcopy(self.population[parents_1][cut_position:])
        return children_1, children_2

    def ga_crossover_arithmetic(self, parents_1, parents_2):
        children_1 = []
        children_2 = []
        for i in range(self.n_variables):
            if type(self.population[parents_1][i]) == float:
                chs1 = self.population[parents_1][i] * self.weight_crossover \
                       + (1 - self.weight_crossover) * self.population[parents_2][i]
                chs2 = self.population[parents_2][i] * self.weight_crossover \
                       + (1 - self.weight_crossover) * self.population[parents_1][i]
                children_1.append(deepcopy(chs1))
                children_2.append(deepcopy(chs2))
            elif type(self.population[parents_1][i]) == str:
                chs1 = self.population[parents_1][i]
                chs2 = self.population[parents_2][i]
                children_1.append(chs1)
                children_2.append(chs2)
            elif type(self.population[parents_1][i]) == list:
                sub_chs1 = []
                sub_chs2 = []
                for k in range(len(self.population[parents_1][i])):
                    sub_chs1.append(self.population[parents_1][i][k] * self.weight_crossover + \
                                    (1 - self.weight_crossover) * self.population[parents_2][i][k])
                    sub_chs2.append(self.population[parents_2][i][k] * self.weight_crossover + \
                                    (1 - self.weight_crossover) * self.population[parents_1][i][k])
                children_1.append(deepcopy(sub_chs1))
                children_2.append(deepcopy(sub_chs2))
            elif type(self.population[parents_1][i]) == int:
                chs1 = self.population[parents_1][i]
                chs2 = self.population[parents_2][i]
                children_1.append(chs1)
                children_2.append(chs2)
            else:
                print('No type found')
        return children_1, children_2

    def ga_mutation(self, mut_descent, mutate):
        if not mutate:
            return mut_descent
        else:
            mutated_probability = []
            mutated_positions = []
            vector = []
            for sublist in self.population[0]:
                if type(sublist) == list:
                    mutated_probability.append(np.random.uniform(low=0, high=1, size=len(sublist)))
                    temp = self.mutation_probability > mutated_probability[-1]
                    mutated_positions.append(temp)
                    vector += list(temp)
                else:
                    mutated_probability.append(np.random.uniform(low=0, high=1))
                    temp = self.mutation_probability > mutated_probability[-1]
                    mutated_positions.append(temp)
                    vector.append(temp)
            if np.all(~np.array(vector)):
                return mut_descent
            else:
                for j in range(len(mutated_positions)):
                    sub_mutated = mutated_positions[j]
                    if self.range_variables[j][0] == 'float_iter':
                        for i in range(len(sub_mutated)):
                            if sub_mutated[i]:
                                mut_descent[j][i] += 0.3 * mut_descent[j][i] * np.random.normal(0, 0.1)
                                if mut_descent[j][i] < self.range_variables[j][1]:
                                    mut_descent[j][i] = self.range_variables[j][1]
                                elif mut_descent[j][i] > self.range_variables[j][2]:
                                    mut_descent[j][i] = self.range_variables[j][2]
                                else:
                                    mut_descent[j][i] = mut_descent[j][i]
                    elif self.range_variables[j][0] == 'float':
                        mut_descent[j] += 0.3 * mut_descent[j] * np.random.normal(0, 0.1)
                        if mut_descent[j] < self.range_variables[j][1]:
                            mut_descent[j] = self.range_variables[j][1]
                        elif mut_descent[j] > self.range_variables[j][2]:
                            mut_descent[j] = self.range_variables[j][2]
                        else:
                            mut_descent[j] = mut_descent[j]
                    elif self.range_variables[j][0] == 'str':
                        pass
                    else:
                        pass
        return mut_descent

    def print_report(self, Hf, Vf, Mf, Tf, individual):
        print(':::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Max thrust by engine: ', individual[-1], ' [N]')
        print('Used fuel mass: ', Mf[0] - Mf[-1], ' [kg]')
        print('-------------------------------------------------')
        print('Switchinf functions')
        ini_best_individuals = [np.array(np.abs(Hf - i)).argmin() for i in individual[4]]
        end_best_individuals = [int(i + individual[1] / self.ga_dynamics.step_width) for i in individual]

        for i in range(len(individual[3])):
            print('sf_', i + 1, ': ', 'H_i = ', Hf[ini_best_individuals[i]], 'H_f = ', Hf[end_best_individuals[i]],
                  'V_i = ', -Vf[ini_best_individuals[i]], 'V_f = ', -Vf[end_best_individuals[i]])
        return

