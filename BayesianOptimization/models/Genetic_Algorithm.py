from typing import Tuple
from models.model_interface import ModelInterface
import deminf_data
import pygad
from tqdm.notebook import tqdm
from models.error import StopModel


class GeneticAlgorithm(ModelInterface):

    def __init__(self, total, progress_bar=None, verbose=True):
        objective1 = deminf_data.Objective.from_name('1_Bot_4_Sim', negate=True, type_of_transform='logarithm')
        genes_space = []
        for l, r in zip(objective1.lower_bound, objective1.upper_bound):
            genes_space.append({'low': l, 'high': r})

        def fitness_func(solution, solution_idx):
            fitness_func.count += 1
            fitness_func.X.append(solution)
            y = objective1(solution)
            if len(fitness_func.Y_best) == 0:
                fitness_func.Y_best.append(y)
            else:
                fitness_func.Y_best.append(min(fitness_func.Y_best[-1], y))
            progress_bar.update(1)
            if fitness_func.count == total:
                raise StopModel('nothing series, just stop of the model')
            return -y

        fitness_func.count = 0
        fitness_func.X = []
        fitness_func.Y_best = []
        if verbose:
            if progress_bar is not None:
                fitness_func.progress_bar = progress_bar
            else:
                fitness_func.progress_bar = tqdm(total=total, desc='GeneticAlgorithm')
        self.fitness_function = fitness_func

        num_generations = 100  # Number of generations.
        num_genes = len(genes_space)
        sol_per_pop = 50

        def callback_generation(ga_instance):
            pass

        # Creating an instance of the GA class inside the ga module.
        # Some parameters are initialized within the constructor.
        self.ga_instance = ga_instance = pygad.GA(num_generations=num_generations,
                                                  num_parents_mating=5,
                                                  fitness_func=self.fitness_function,
                                                  sol_per_pop=sol_per_pop,
                                                  num_genes=num_genes,
                                                  gene_space=genes_space,
                                                  on_generation=callback_generation)

    def fit(self) -> Tuple[int, list]:
        try:
            self.ga_instance.run()
        except Exception as exc:
            if not isinstance(exc, StopModel):
                raise exc
        return self.fitness_function.count, self.fitness_function.Y_best
