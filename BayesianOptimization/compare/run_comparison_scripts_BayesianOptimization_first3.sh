#!/usr/bin/env bash

objective_names=(1_Bot_4_Sim 2_ExpDivNoMig_5_Sim 2_DivMig_5_Sim)

for i in "${objective_names[@]}"; do
  echo "BayesianOptimization on $i is being executed"
  python3 -m compare.Comparison_script_BayesianOptimization "${i}" 50 300
done



