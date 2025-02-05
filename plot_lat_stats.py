'''
Feito pra plotar a variacao dos parametros estatisticos de forma a ver a evolução latitudinal deles
 Principalmente skill e correlação
 Isso vai definir o modelo com melhor desempenho, que vai ser utilizado nas análises de wavelet, hovmoller e corrente.
'''

import pandas as pd
from matplotlib import pyplot as plt
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
            'PORTO DO FORNO': (-22.9675, -42.00222222222222)}
    
    return coords[point]

# Importar as métricas
metric_path = '/Users/breno/mestrado/n_stats/'
skill_path = '/Users/breno/mestrado/skills/'

# Atribuir as latitudes aos pontos
skills = {}
models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'HYCOM', 'ORAS']
for i in os.listdir(skill_path):
    if i[0] == '.':
        continue
    skills[i] = ({}, get_coords(i))
    for model in models:
        skills[i][0][model] = pd.read_csv(skill_path + i + f'/{model}_skills.csv', header=None)


# organizar os dados
# quero pegar os dados já organizados de sul pra norte
ss1 = {}
ss2 = {}
ss3 = {}
for model in models:
    ss1[model] = {}
    ss2[model] = {}
    ss3[model] = {}
    for place in skills:
        ss1[model][skills[place][1][0]] = skills[place][0][model][1][0]
        ss2[model][skills[place][1][0]] = skills[place][0][model][1][1]
        ss3[model][skills[place][1][0]] = skills[place][0][model][1][2]


        ss1[model]= dict(sorted(ss1[model].items()))
        ss2[model]= dict(sorted(ss2[model].items()))
        ss3[model]= dict(sorted(ss3[model].items()))


# Fazer o plot

# fig, axes = plt.subplots(1,3, figsize=(15, 10))

# for model in models:
#     lats = list(ss1[model].keys())
#     values1 = list(ss1[model].values())
#     values2 = list(ss2[model].values())
#     values3 = list(ss3[model].values())

#     if model == 'HYCOM':
#         model = 'GOFS'

#     axes[0].plot(values1, lats, label=model, marker='o', linestyle='')
#     axes[0].set_xlim(0,1)
#     axes[0].set_ylim(-35,-20)
#     axes[0].set_title('skill 1')


#     axes[1].plot(values2, lats, marker='o', linestyle='')
#     axes[1].set_xlim(0,1)
#     axes[1].set_ylim(-35,-20)
#     axes[1].set_title('skill 2')


#     axes[2].plot(values3, lats, marker='o', linestyle='')
#     axes[2].set_xlim(0,1)
#     axes[2].set_ylim(-35,-20)
#     axes[2].set_title('skill 3')


# fig.legend(models, loc='upper center', ncol=len(models))

# plt.tight_layout()
# path = '/Users/breno/mestrado/lat_skills_s20.png'
# plt.savefig(path)

models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'HYCOM', 'ORAS']


fig, axes = plt.subplots(1,1, figsize=(5, 10))

for model in models:
    lats = list(ss1[model].keys())
    values1 = list(ss1[model].values())
    values2 = list(ss2[model].values())
    values3 = list(ss3[model].values())

    if model == 'HYCOM':
        model = 'GOFS'

    axes.plot(values2, lats, label=model, marker='o', linestyle='')
    axes.set_xlim(0,1)
    # axes[0].set_ylim(-35,-20)
    # axes[0].set_title('')


    # axes[1].plot(values2, lats, marker='o', linestyle='')
    # axes[1].set_xlim(0,1)
    # axes[1].set_ylim(-35,-20)
    # axes[1].set_title('skill 2')


    # axes[2].plot(values3, lats, marker='o', linestyle='')
    # axes[2].set_xlim(0,1)
    # axes[2].set_ylim(-35,-20)
    # axes[2].set_title('skill 3')

models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'GOFS', 'ORAS']

fig.legend(models)

plt.tight_layout()
path = '/Users/breno/mestrado/lat_skills2.png'
plt.savefig(path)


#######

models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'HYCOM', 'ORAS']

fig, axes = plt.subplots(2, 1, figsize=(5, 10))  # 2 linhas, 1 coluna

# Primeiro subplot (latitudes até -20)
for model in models:
    lats = list(ss1[model].keys())
    values1 = list(ss1[model].values())
    values2 = list(ss2[model].values())
    values3 = list(ss3[model].values())

    if model == 'HYCOM':
        model = 'GOFS'

    axes[1].plot(values2, lats, label=model, marker='o', linestyle='')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-35, -20)  # Limite do eixo y para o primeiro subplot
    axes[1].set_xlabel('Skill')
    axes[1].set_ylabel('Latitude')

# Segundo subplot (latitudes a partir de -5)
for model in models:
    lats = list(ss1[model].keys())
    values1 = list(ss1[model].values())
    values2 = list(ss2[model].values())
    values3 = list(ss3[model].values())

    if model == 'HYCOM':
        model = 'GOFS'

    axes[0].plot(values2, lats, label=model, marker='o', linestyle='')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-5, 0)  # Limite do eixo y para o segundo subplot
    axes[0].set_ylabel('Latitude')

models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'GOFS', 'ORAS']


# Adicionando a legenda
fig.legend(models, loc='upper right')

# Ajustando o layout
plt.tight_layout()

# Exibindo a figura
plt.show()


##############

models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'HYCOM', 'ORAS']

# Criando a figura com dois subplots e definindo a proporção de altura
fig, axes = plt.subplots(2, 1, figsize=(5, 10), gridspec_kw={'height_ratios': [1, 8]})  # 2 linhas, 1 coluna

# Primeiro subplot (latitudes de -5 a -3)
for model in models:
    lats = np.asarray(list(ss1[model].keys()))
    values2 = np.asarray(list(ss2[model].values()))

    lats = lats[values2 > 0.001]
    values2 = values2[values2 > 0.001]

    if model == 'HYCOM':
        model = 'GOFS'

    axes[0].plot(values2, lats, label=model, marker='o', linestyle='')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-5, -3)  # Limite do eixo y para o primeiro subplot
    axes[0].set_xticks([])
    # axes[0].set_ylabel('Latitude')

# Segundo subplot (latitudes de -35 a -20)
for model in models:
    lats = np.asarray(list(ss1[model].keys()))
    values2 = np.asarray(list(ss2[model].values()))

    lats = lats[values2 > 0.001]
    values2 = values2[values2 > 0.001]

    if model == 'HYCOM':
        model = 'GOFS'

    axes[1].plot(values2, lats, label=model, marker='o', linestyle='')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-35, -20)  # Limite do eixo y para o segundo subplot
    axes[1].set_xlabel('Skill')
    axes[1].set_ylabel('Latitude')



# Adicionando a legenda
models = ['BRAN', 'CGLO', 'FOAM', 'GLOR4', 'GLOR12', 'GOFS', 'ORAS']
fig.legend(models)

# Ajustando o layout para aproximar os subplots
plt.subplots_adjust(hspace=0.1)  # Espaçamento vertical entre os subplots

# Exibindo a figura
path = '/Users/breno/mestrado/lat_skills_t_dropna.png'
plt.tight_layout()
plt.savefig(path)


###############

# fazendo uma média de skill por latitude:

lats = np.asarray(list(ss1['BRAN'].keys()))

lat_mean = {}
for lat in lats:
    mean_lat = []
    for model in models:
        mean_lat.append(ss2[model][lat])
    mean = np.asarray(mean_lat).mean()
    lat_mean[lat] = mean


plt.plot(list(lat_mean.values()), list(lat_mean.keys()))


# skill medio por modelo:

for model in models:
    print(model)
    print(np.nanmean(np.asarray(list(ss2[model].values()))))
    print('------------------------------')