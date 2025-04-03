import pandas as pd
import numpy as np
import os

def get_coords(point):


    coords = {'PORTO DE MUCURIPE': (-3.7025, -38.46833333333333),
            'PAGA DÍVIDAS I': (-1.2016666666666669, -52.11833333333333),
            'Salvador_glossbrasil': (-12.97 + 0.3, -38.52),# mudando pq tem 3 salvador
            'CAPITANIA DE SALVADOR': (-12.967777777777778 + 0.2, -38.516666666666666), # mudando pq tem 3 salvador
            'Salvador': (-12.97, -38.52),
            'CANIVETE': (0.5013888888888889, -50.40222222222222),
            'PORTO DE PARANAGUÁ - CAIS OESTE': (-25.50027777777778, -48.51805555555556),
            'BARRA DE PARANAGUÁ - CANAL DA GALHETA': (-25.566666666666663,
            -48.31666666666667),
            'CIA DOCAS DO PORTO DE SANTANA': (-0.0519444444444444, -51.16694444444444),
            'Ubatuba_gloss': (-23.5 + 0.1, -45.12), # mudando pq tem 2 ubatuba
            'TIPLAM': (-23.86777777777778, -46.367777777777775),
            'salvador': (-12.97  + 0.1, -38.52),  # mudando pq tem 3 salvador
            'Fortaleza': (-3.72, -38.47),
            'rio_grande': (-32.13, -52.1),
            'SERRARIA RIO MATAPI': (-0.0344444444444444, -51.20027777777778),
            'Imbituba': (-28.13, -48.4),
            'PORTO DE VILA DO CONDE': (-1.5344444444444445, -48.75055555555556),
            'IGARAPÉ GRD DO CURUÁ': (0.7522222222222222, -50.11666666666667),
            'TERMINAL PORTUÁRIO DA PONTA DO FÉLIX': (-25.45083333333333,
            -48.66861111111111),
            'ubatuba': (-23.5, -45.12),
            'TEPORTI': (-26.86833333333333, -48.70194444444445),
            'ilha_fiscal': (-22.9, -43.17),
            'Macae': (-22.23, -41.47),
            'NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ': (-26.901666666666664,
            -48.65055555555556),
            'PORTO DO FORNO': (-22.9675, -42.00222222222222),
            'Porto do forno': (-22.9675, -42.00222222222222)}
    
    return coords[point]


models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'HYCOM', 'ORAS']
places = os.listdir('/Users/breno/mestrado/comp_series3/')



# Atribuir as latitudes aos pontos
skills = {}
for place in places:
    if place == 'CAPITANIA DE SALVADOR':
        continue
    if place[0] == '.':
        continue
    skill_path = f'/Users/breno/mestrado/comp_series3/{place}/stats/'
    skills[place] = ({}, get_coords(place))
    for model in models:
        skills[place][0][model] = pd.read_csv(skill_path + f'/{model}skills.csv', header=None)

# ss1 = {}
# ss2 = {}
# ss3 = {}
# for model in models:

#     ss1[model] = {}
#     ss2[model] = {}
#     ss3[model] = {}
#     for place in skills:
#         ss1[model][skills[place][1][0]] = skills[place][0][model][1][0]
#         ss2[model][skills[place][1][0]] = skills[place][0][model][1][1]
#         ss3[model][skills[place][1][0]] = skills[place][0][model][1][2]


#         ss1[model]= dict(sorted(ss1[model].items()))
#         ss2[model]= dict(sorted(ss2[model].items()))
#         ss3[model]= dict(sorted(ss3[model].items()))

####
# second approach

ss1 = {}
ss2 = {}
ss3 = {}
for model in models:

    ss1[model] = {}
    ss2[model] = {}
    ss3[model] = {}
    for place in skills:
        ss1[model][place] = skills[place][0][model][1][0]
        ss2[model][place] = skills[place][0][model][1][1]
        ss3[model][place] = skills[place][0][model][1][2]


        ss1[model]= dict(sorted(ss1[model].items()))
        ss2[model]= dict(sorted(ss2[model].items()))
        ss3[model]= dict(sorted(ss3[model].items()))

df = pd.DataFrame(ss2)

lats = []
for loc in df.index:
    lats.append(np.round(skills[loc][1][0], 2))

df['Lat'] = lats
df = df.sort_values('Lat')



df['Lat. Mean'] = df[['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'HYCOM', 'ORAS']].mean(axis=1) # ta contando a latitude na media, preciso corrigir
df.loc['Mean'] = df.mean(axis=0)



# lats = np.asarray(list(ss1['BRAN'].keys()))

lat_mean = {}
for lat in lats:
    mean_lat = []
    for model in models:
        mean_lat.append(ss2[model][lat])
    mean = np.asarray(mean_lat).mean()
    lat_mean[lat] = mean

means = []
stds = []
for model in models:
    print(model)
    # print(np.mean(np.asarray(list(ss1[model].values()))))
    print(np.mean(np.asarray(list(corr[model].values()))))
    means.append(np.mean(np.asarray(list(corr[model].values()))))
    print(np.std(np.asarray(list(corr[model].values()))))
    stds.append(np.std(np.asarray(list(corr[model].values()))))
    # print(np.mean(np.asarray(list(ss3[model].values()))))
    print('------------------------------')