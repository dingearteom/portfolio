from compare.Draw_comparison import DrawComparison

import sys

key_params = {}

for arv in sys.argv[1:]:
    key, value = arv.split('=')
    if key == 'cut_off':
        value = int(value)
    elif key == 'trajectories':
        value = (value == 'True')
    # key: model is also possible
    key_params[key] = value

objective_names = ['1_Bot_4_Sim', '2_ExpDivNoMig_5_Sim', '2_DivMig_5_Sim']
for objective_name in objective_names:
    model = DrawComparison(objective_name=objective_name, real=False, **key_params)
    if 'trajectories' not in key_params:
        model.draw()
    else:
        if not 'model' in key_params:
            model.draw_trajectories('BayesianOptimization')
            model.draw_trajectories('Gadma')
        else:
            model.draw_trajectories(key_params['model'])
