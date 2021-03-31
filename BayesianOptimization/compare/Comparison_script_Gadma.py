from models.Gadma import Gadma
import tqdm
import pickle
import sys
import os
from datetime import datetime

objective_name = str(sys.argv[-3])
num_run = int(sys.argv[-2])
num_evaluation = int(sys.argv[-1])

progress_bar_Gadma = tqdm.tqdm(total=num_evaluation * num_run, desc='Gadma')
Gadma_Y_best = []

start_time = datetime.now()

for i in range(num_run):
    model = Gadma(total=num_evaluation, objective_name=objective_name, progress_bar=progress_bar_Gadma, run=i+1)
    Y_best = model.fit()
    Gadma_Y_best.append(Y_best)

time_of_execution = datetime.now() - start_time

hours, remainder = divmod(time_of_execution.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)

path_dir = 'compare/data/Y_best'
if not os.path.exists(path_dir):
    os.mkdir(path_dir)

with open(f"compare/data/Y_best/Y_best_Gadma_{objective_name}_saved.pickle", "wb") as fp:
    pickle.dump(Gadma_Y_best, fp)

path_to_file = 'compare/data/log_Y_best.txt'
if not os.path.exists(path_to_file):
    open(path_to_file, 'w').close()
with open(path_to_file, 'a') as fp:
    fp.write(f'Gadma {objective_name} {num_run}x{num_evaluation} {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
             f' execution_time:{"{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))} \n')