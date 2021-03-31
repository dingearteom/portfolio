from models.Genetic_Algorithm import GeneticAlgorithm
import tqdm
import pickle
import sys
import os

num_run = int(sys.argv[-2])
num_evaluation = int(sys.argv[-1])

progress_bar_GeneticAlgorithm = tqdm.tqdm(total=num_evaluation * num_run, desc='GeneticAlgorithm')
GeneticAlgorithm_Y_best = []

for i in range(num_run):
    model = GeneticAlgorithm(num_evaluation, progress_bar_GeneticAlgorithm)
    _, Y_best = model.fit()
    GeneticAlgorithm_Y_best.append(Y_best)

path_dir = 'compare/data/Y_best'
if not os.path.exists(path_dir):
    os.mkdir(path_dir)

with open("compare/data/Y_best/GeneticAlgorithm_saved.pickle", "wb") as fp:
    pickle.dump(GeneticAlgorithm_Y_best, fp)