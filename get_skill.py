'''
Em todos os casos, considera-se que os inputs das funcoes sao arrays
'''
import numpy as np
import pandas as pd
import csv


def ss1(data, model):
    # MURPHY et al. (1995)
    up = (data - model) **2
    bot = (data - data.mean())**2
    ss = 1 - (up.sum()/bot.sum())

    return ss
   

def ss2(data, model):
    # TAYLOR (2001)
    R = np.corrcoef(data, model)[0,1]
    sig = model.std()/data.std()
    up = 4 * (1+R)
    bot = 2 * (sig + 1/sig)**2
    ss = up/bot

    return ss


def ss3(data, model):
    # METZGER et al. (2008)
    R = np.corrcoef(data, model)[0,1]
    t1 = R**2
    t2 = (R - (data.std()/ model.std()))**2
    t3top = data.mean() - model.mean()
    t3bot = model.std()
    t3 = (t3top/t3bot)**2

    ss = t1 - t2 - t3

    return ss


def run_all(data, model, path):
    # roda tudo e exporta num csv
    ss = {}
    ss['ss1'] = ss1(data, model)
    ss['ss2'] = ss2(data, model)
    ss['ss3'] = ss3(data, model)

    (pd.DataFrame.from_dict(data=ss, orient='index')
    .to_csv(path + 'skills.csv', header=False))

    