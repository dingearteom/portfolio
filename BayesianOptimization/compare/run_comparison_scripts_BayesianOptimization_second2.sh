#!/usr/bin/env bash

objective_names=(2_BotDivMig_8_Sim 3_DivMig_8_Sim)

for i in "${objective_names[@]}"; do
  echo "BayesianOptimization on $i is being executed"
  python3 -m compare.Comparison_script_BayesianOptimization "${i}" 50 300
done



