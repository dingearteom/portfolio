import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import os

class DrawComparison:
    def __init__(self, objective_name, real, cut_off=0, **kwargs):
        self.objective_name = objective_name
        self.BayesianOptimization_data_file_name = f'Y_best_BayesianOptimization_{objective_name}_saved'
        self.GeneticAlgorithm_data_file_name = f'Y_best_Gadma_{objective_name}_saved'
        self.picture_file_name = f'comparison_{objective_name}'
        if cut_off != 0:
            self.picture_file_name += f'_cut_off{cut_off}'
        self.real = real
        self.cut_off = cut_off

        with open(f'compare/data/Y_best/{self.BayesianOptimization_data_file_name}.pickle', 'rb') as fp:
            Y_best = pickle.load(fp)

        self.num_run = len(Y_best)
        self.num_evaluation = len(Y_best[0])

        with open(f'compare/data/Y_best/{self.GeneticAlgorithm_data_file_name}.pickle', 'rb') as fp:
            Y_best = pickle.load(fp)

        assert self.num_run == len(Y_best) and self.num_evaluation == len(Y_best[0]), 'num_run and num_evaluation ' \
                                                                                      'are not the same for' \
                                                                                      ' BayesianOptimization' \
                                                                                      'and Gadma. BayesianOptimization'

    def draw(self, alpha=0.5):
        ax = plt.subplot(111)
        self._draw('BayesianOptimization', alpha)
        self._draw('Gadma', alpha)
        plt.legend()

        if self.real:
            pictures_dir = 'compare/data/pictures/real'
            pictures_log_dir = 'compare/data/pictures_log/real'
        else:
            pictures_dir = 'compare/data/pictures/artificial'
            pictures_log_dir = 'compare/data/pictures_log/artificial'
        if self.cut_off == 0:
            pictures_dir = f'{pictures_dir}/main'
            pictures_log_dir = f'{pictures_log_dir}/main'
        else:
            pictures_dir = f'{pictures_dir}/modified'
            pictures_log_dir = f'{pictures_log_dir}/modified'
        if not os.path.exists(pictures_dir):
            os.makedirs(pictures_dir)
        if not os.path.exists(pictures_log_dir):
            os.makedirs(pictures_log_dir)
        with open(f'{pictures_log_dir}/{self.picture_file_name}.pickle', 'wb') as fp:
            pickle.dump(ax, fp)
        plt.savefig(f'{pictures_dir}/{self.picture_file_name}.png')
        plt.clf()

    def _draw(self, model, alpha):
        if model == 'BayesianOptimization':
            with open(f'compare/data/Y_best/{self.BayesianOptimization_data_file_name}.pickle', 'rb') as fp:
                Y_best = pickle.load(fp)
        elif model == 'Gadma':
            with open(f'compare/data/Y_best/{self.GeneticAlgorithm_data_file_name}.pickle', 'rb') as fp:
                Y_best = pickle.load(fp)

        num_run = len(Y_best)
        num_evaluation = len(Y_best[0])

        def get_confidence_interval(arr, alpha):
            arr = sorted(arr)
            n = len(arr)
            return arr[int((1 - alpha) / 2) * n], arr[min(math.ceil(((1 + alpha) / 2) * n), n - 1)]

        Y_best = np.array(Y_best)
        Y_best_mean = np.median(Y_best, axis=0)

        Y_best_range_low = []
        Y_best_range_high = []
        for i in range(num_evaluation):
            low, high = get_confidence_interval(Y_best[:, i].flatten(), alpha=alpha)
            Y_best_range_low.append(low)
            Y_best_range_high.append(high)

        color = ('blue' if model == 'Gadma' else 'orange')
        plt.plot(range(self.cut_off + 1, num_evaluation + 1), Y_best_mean[self.cut_off:],
                 label=f'{model}', color=color)
        plt.fill_between(range(self.cut_off + 1, num_evaluation + 1),
                         Y_best_range_low[self.cut_off:],
                         Y_best_range_high[self.cut_off], alpha=0.2, color=color)

    def draw_trajectories(self, model, alpha=0.5):
        '''

        :param model: string, Gadma or BayesianOptimization
        :return:
        '''
        assert model == 'Gadma' or model == 'BayesianOptimization', f'Unknown model name {model}. ' \
                                                                    f'Only Gadma or BayesianOptimization are allowed'
        log_dir = f'compare/data/{model}_log/{self.objective_name}'
        ax = plt.subplot(111)
        self._draw(model, alpha)

        for i in range(1, self.num_run + 1):
            with open(f'{log_dir}/log_{i}.pickle', 'rb') as fp:
                X = pickle.load(fp)
                Y = pickle.load(fp)
                Y_best = pickle.load(fp)
                Y_best = Y_best[:self.num_evaluation]
                color = ('blue' if model=='Gadma' else 'orange')
                plt.plot(range(self.cut_off + 1, self.num_evaluation + 1), Y_best[self.cut_off:], color=color)
        plt.legend()

        if self.real:
            pictures_dir = 'compare/data/pictures/real'
            pictures_log_dir = 'compare/data/pictures_log/real'
        else:
            pictures_dir = 'compare/data/pictures/artificial'
            pictures_log_dir = 'compare/data/pictures_log/artificial'
        if self.cut_off == 0:
            pictures_dir = f'{pictures_dir}/main'
            pictures_log_dir = f'{pictures_log_dir}/main'
        else:
            pictures_dir = f'{pictures_dir}/modified'
            pictures_log_dir = f'{pictures_log_dir}/modified'

        pictures_dir = f'{pictures_dir}/trajectories/{self.objective_name}'
        pictures_log_dir = f'{pictures_log_dir}/trajectories/{self.objective_name}'
        if not os.path.exists(pictures_dir):
            os.makedirs(pictures_dir)
        if not os.path.exists(pictures_log_dir):
            os.makedirs(pictures_log_dir)

        file_name = f'{model}'
        if self.cut_off != 0:
            file_name += f'_cut_off{self.cut_off}'

        with open(f'{pictures_log_dir}/{file_name}.pickle', 'wb') as fp:
            pickle.dump(ax, fp)
        plt.savefig(f'{pictures_dir}/{file_name}.png')
        plt.clf()
