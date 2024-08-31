'''
POINT/LAT / MODEL
##########/ COR. / RMSE / STD
ILHA FISCAL / -23 

'''
import os
import pandas as pd
from easy_mpl import taylor_plot
from matplotlib import pyplot as plt

models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'HYCOM', 'ORAS']
# path_stats = '/Users/breno/Documents/Mestrado/resultados/2012/figs/stats/'
path_stats = '/Users/breno/mestrado/CPAM/stats/'
simulations = {}
for point in os.listdir(path_stats):
    if point[0] == '.':
        continue
    simulations[point] = {}
    print(point)
    for model in models:
        comp_stats = pd.read_csv(path_stats + point + '/comp_' + model + '.csv')
        corr = comp_stats['0'][0]
        nbias = comp_stats['0'][1]
        std = pd.read_csv(path_stats + point + '/gen_' + model + '.csv')['0'][2]
        simulations[point][model] = {'corr_coeff':corr/100, 'pbias':nbias, 'std':std}
        print(model + ' : ' + str(nbias))
    
    observations = {'std': pd.read_csv(path_stats + point + '/gen_' + point + '.csv')['0'][2]}
    predictions = simulations[point]
    fig = taylor_plot(observations, predictions, true_label ='Dado', title=f"{point}", show=False, lang='pt')
    fig.savefig(f'/Users/breno/mestrado/CPAM/taylor/{point}.png')
    # fig.savefig(f'/Users/breno/Documents/Mestrado/resultados/2012/figs/taylor/{point[:-4]}.png')



>>> observations = {'std': 4.916}
>>> predictions = {   # pbias is optional
...         'Model 1': {'std': 2.80068, 'corr_coeff': 0.49172, 'pbias': -8.85},
...         'Model 2': {'std': 3.47, 'corr_coeff': 0.67, 'pbias': -19.76},
...         'Model 3': {'std': 3.53, 'corr_coeff': 0.596, 'pbias': 7.81},
...         'Model 4': {'std': 2.36, 'corr_coeff': 0.27, 'pbias': -22.78},
...         'Model 5': {'std': 2.97, 'corr_coeff': 0.452, 'pbias': -7.99}}
...
>>> taylor_plot(observations,
...     predictions,
...     title="with statistical parameters")


'''
>>> import numpy as np
>>> from easy_mpl import taylor_plot
>>> np.random.seed(313)
>>> observations =  np.random.normal(20, 40, 10)
>>> simulations =  {"LSTM": np.random.normal(20, 40, 10),
...             "CNN": np.random.normal(20, 40, 10),
...             "TCN": np.random.normal(20, 40, 10),
...             "CNN-LSTM": np.random.normal(20, 40, 10)}
>>> taylor_plot(observations=observations,
...             simulations=simulations,
...             title="Taylor Plot")
'''

'''
>>> observations = {'std': 4.916}
>>> predictions = {   # pbias is optional
...         'Model 1': {'std': 2.80068, 'corr_coeff': 0.49172, 'pbias': -8.85},
...         'Model 2': {'std': 3.47, 'corr_coeff': 0.67, 'pbias': -19.76},
...         'Model 3': {'std': 3.53, 'corr_coeff': 0.596, 'pbias': 7.81},
...         'Model 4': {'std': 2.36, 'corr_coeff': 0.27, 'pbias': -22.78},
...         'Model 5': {'std': 2.97, 'corr_coeff': 0.452, 'pbias': -7.99}}
...
>>> taylor_plot(observations,
...     predictions,
...     title="with statistical parameters")
...
... # with customized markers
>>> np.random.seed(313)
>>> observations =  np.random.normal(20, 40, 10)
>>> simulations =  {"LSTM": np.random.normal(20, 40, 10),
...                 "CNN": np.random.normal(20, 40, 10),
...                 "TCN": np.random.normal(20, 40, 10),
...                 "CNN-LSTM": np.random.normal(20, 40, 10)}
>>> taylor_plot(observations=observations,
...             simulations=simulations,
...             title="customized markers",
...             marker_kws={'markersize': 10, 'markeredgewidth': 1.5,
...                 'markeredgecolor': 'black'})
...
... # with customizing bbox
>>> np.random.seed(313)
>>> observations =  np.random.normal(20, 40, 10)
>>> simus =  {"LSTMBasedRegressionModel": np.random.normal(20, 40, 10),
...         "CNNBasedRegressionModel": np.random.normal(20, 40, 10),
...         "TCNBasedRegressionModel": np.random.normal(20, 40, 10),
...         "CNN-LSTMBasedRegressionModel": np.random.normal(20, 40, 10)}
>>> taylor_plot(observations=observations,
...             simulations=simus,
...             title="custom_legend",
...             leg_kws={'facecolor': 'white',
...                 'edgecolor': 'black','bbox_to_anchor':(1.1, 1.05)})
'''