from models.model_interface import ModelInterface
import GPyOpt
from typing import Tuple
import deminf_data
import tqdm
import tqdm.notebook
import os
import pickle


class BayesianOptimization(ModelInterface):

    def __init__(self, total, objective_name, progress_bar=None, verbose=True, notebook=True, log=True, run=None):
        self.num_iter = total
        self.log = log
        if run is not None:
            self.run = run
        else:
            self.run = 'test_run'
        objective = deminf_data.Objective.from_name(objective_name, negate=True, type_of_transform='logarithm')
        bounds = []
        for i, (l, r) in enumerate(zip(objective.lower_bound, objective.upper_bound), start=1):
            bounds.append({'name': f'var_{i}', 'type': 'continuous', 'domain': (l, r)})

        def f(x):
            x = x.copy()
            Y = []
            for point in x:
                f.count += 1
                f.X.append(point)
                y = objective(point)
                f.Y.append(y)
                if len(f.Y_best) == 0:
                    f.Y_best.append(y)
                else:
                    f.Y_best.append(min(f.Y_best[-1], y))

                if verbose:
                    f.progress_bar.update(1)
                Y.append(y)
            return Y

        f.count = 0
        f.X = []
        f.Y = []
        f.Y_best = []
        if verbose:
            if progress_bar is not None:
                f.progress_bar = progress_bar
            else:
                if notebook:
                    f.progress_bar = tqdm.notebook.tqdm(total=total, desc='BayesianOptimization')
                else:
                    f.progress_bar = tqdm.tqdm(total=total, desc='BayesianOptimization')

        self.f = f
        self.objective_name = objective_name
        self.model = GPyOpt.methods.BayesianOptimization(f=f,
                                                         domain=bounds,
                                                         model_type='GP',
                                                         acquisition_type='EI',
                                                         normalize_Y=True,
                                                         acquisition_weight=2,
                                                         model_update_interval=1,
                                                         verbosity=True)

    def fit(self) -> Tuple[int, list]:
        self.model.run_optimization(self.num_iter, eps=-1)
        if self.log:
            self.write_log()
        return self.f.count, self.f.Y_best

    def write_log(self):
        path_to_file = f'compare/data/BayesianOptimization_log/{self.objective_name}/log_{self.run}.pickle'
        path_to_dir = 'compare/data/BayesianOptimization_log'
        if not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)
        path_to_dir = f'compare/data/BayesianOptimization_log/{self.objective_name}'
        if not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)
        open(path_to_file, 'w')
        with open(path_to_file, 'wb') as fp:
            pickle.dump(self.f.X, fp)
            pickle.dump(self.f.Y, fp)
            pickle.dump(self.f.Y_best, fp)