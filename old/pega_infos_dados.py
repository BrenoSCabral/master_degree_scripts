import pandas as pd
import re
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

infos = '/Users/breno/Downloads/Inventario alturas cadastradas.pdf'

from pdfminer.high_level import extract_text

text = extract_text(infos)

padrao_estacao = re.compile(r'[\d]+ - [\D]+')
# nao pega a estacao 10687 - 5-MARPEX V_PONTO B_28MAR A 01MAI23_1MIN_BRUTO pg 114

padrao_dias = re.compile(r'[\d]{2}\/[\d]{2}\/[\d]{4}[\d]{2}\/[\d]{2}\/[\d]{4}[\d]+')

padrao_localidade = re.compile(r'W[\S]+ÁREA')
padrao_localidade.findall(text)

pages = text.split('MARINHA DO BRASILDIRETORIA DE HIDROGRAFIA E NAVEGAÇÃOCENTRO DE HIDROGRAFIA DA MARINHA INVENTÁRIO DE ALTURAS CADASTRADASPágina ')


# Dicionário para armazenar os dados de cada estação
dados_estacoes = {}
# dias_regioes = {}
total_dias = []
series_longas = []
series_medias = []

for page in pages:
    try:
        estacao = padrao_estacao.findall(page)
        estacao = estacao[0]

        localidade = padrao_localidade.findall(page)
        localidade = localidade[0][1:-4]

    except:
        continue

    dias = padrao_dias.findall(page)

    n_dias_total = []
    data_inicial = []
    data_final = []
    for i in dias:
        di = datetime.strptime(i[:10], '%d/%m/%Y')
        df = datetime.strptime(i[10:20], '%d/%m/%Y')
        len_dias = len(str((df - di).days))
        n_dias = i[20:20+len_dias]
        n_dias = int(n_dias)

        if n_dias >179:
            series_medias.append((estacao, di, df))
        if n_dias > 364:
            series_longas.append((estacao, di, df))

        n_dias_total.append(n_dias)

        total_dias.append(n_dias)

        data_inicial.append(di)

        data_final.append(df)

    # dias_regioes[localidade] = n_dias_total

    # Armazenar os dados no dicionário de estação
    dados_estacoes.setdefault(estacao, []).extend(list(zip(n_dias_total, data_inicial, data_final)))

# Criar uma lista de tuplas contendo o nome da estação e os dados correspondentes
dados_concatenados = [(estacao, *dado) for estacao, dados in dados_estacoes.items()  for dado in dados]
# asterisco pra incluir so os itens da tupla

# Criar MultiIndex a partir dos dados da estação
multi_index = pd.MultiIndex.from_tuples(dados_concatenados, names=['Estação', 'No Dias', 'Data Inicial', 'Data Final'])

# Criar DataFrame a partir do MultiIndex
df = pd.DataFrame(index=multi_index)

# plotando

# Plotar histograma
 
arr_dias = np.asarray(total_dias)
# Filtrar os valores menores que 15
d15 = len(arr_dias[arr_dias < 15])

# Filtrar os valores entre 15 e 30
d15_30 = len(arr_dias[(arr_dias >= 15) & (arr_dias < 30)])

# Filtrar os valores entre 30 e 60
d30_60 = len(arr_dias[(arr_dias >= 30) & (arr_dias < 60)])

# Filtrar os valores entre 60 e 180
d60_180 = len(arr_dias[(arr_dias >= 60) & (arr_dias < 180)])

d365= len(arr_dias[(arr_dias >= 180) & (arr_dias < 365)])

# Filtrar os valores maiores que 364
dmaior_364 = len(arr_dias[arr_dias > 364])


hist_dias = [d15, d15_30, d30_60, d60_180, d365, dmaior_364]


plt.bar(['<15', '<30', '<60', '<180','<365' ,'>=365'], hist_dias)
plt.title('Tempo de Coleta Ininterrupta')
plt.xlabel('Dias')
plt.ylabel('Ocorrências')

plt.grid()



caminho_arquivo = 'series_longas.txt'

# Abrir o arquivo de texto em modo de escrita ('w')
with open(caminho_arquivo, 'w') as arquivo:
    # Escrever cada elemento da lista em uma linha do arquivo
    for item in series_longas:

        arquivo.write(f"{item[0]}: {datetime.strftime(item[1], '%d/%m/%Y')} - {datetime.strftime(item[2], '%d/%m/%Y')}\n")


caminho_arquivo = 'series_medias.txt'
with open(caminho_arquivo, 'w') as arquivo:
    for item in series_medias:

        arquivo.write(f"{item[0]}: {datetime.strftime(item[1], '%d/%m/%Y')} - {datetime.strftime(item[2], '%d/%m/%Y')}\n")



fig, ax = plt.subplots()

# Plotar as linhas horizontais para cada item da lista
for i, (nome, data_inicio, data_fim) in enumerate(series_longas):
    ax.hlines(y=i, xmin=data_inicio, xmax=data_fim, color='blue')
    # ax.text(data_inicio, i, nome, ha='right', va='center')  # Adiciona o nome do item próximo ao início da linha

# Configurar o eixo y
# ax.set_yticks(range(len(series_longas)))
# ax.set_yticklabels([nome for nome, _, _ in series_longas])

# Definir os rótulos dos eixos
ax.set_xlabel('Data')

# Definir o título do gráfico
ax.set_title('Período de Tempo Séries Longas')

# Rotacionar os rótulos do eixo x para melhorar a legibilidade
plt.xticks(rotation=45)

# Exibir o gráfico
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()

# Plotar as linhas horizontais para cada item da lista
for i, (nome, data_inicio, data_fim) in enumerate(series_medias):
    ax.hlines(y=i, xmin=data_inicio, xmax=data_fim, color='blue')
    # ax.text(data_inicio, i, nome, ha='right', va='center')  # Adiciona o nome do item próximo ao início da linha

# Configurar o eixo y
# ax.set_yticks(range(len(series_longas)))
# ax.set_yticklabels([nome for nome, _, _ in series_longas])

# Definir os rótulos dos eixos
ax.set_xlabel('Data')

# Definir o título do gráfico
ax.set_title('Período de Tempo Séries Médias')

# Rotacionar os rótulos do eixo x para melhorar a legibilidade
plt.xticks(rotation=45)

# Exibir o gráfico
plt.tight_layout()
plt.show()

# REGIONALIZANDO
# ABANDONADO POIS ELES N COLOCAM DIREITINHO O ESTADO, MAS SIM NOME DA CIDADE OU OUTRO PARAMETRO QUALQUER


# N = ['TOCANTINS', 'PARÁ', 'AMAPÁ', 'AMAZONAS', 'RORAIMA', 'RONDÔNIA', 'ACRE']
# NE = ['BAHIA', 'SERGIPE', 'ALAGOAS', 'PERNAMBUCO', 'PARAIBA', 'RIO GRANDE DO NORTE', 'CEARA', 'PIAUI', 'MARANHAO'] 
# SE = ['MINAS GERAIS', 'ESPÍRITO SANTO', 'RIO DE JANEIRO', 'SÃO PAULO']
# S = ['PARANÁ', 'SANTA CATARINA', 'RIO GRANDE DO SUL']

# dias_regionalizados = {'N': [], 'NE': [], 'SE': [], 'S': [], 'OUTRO': []}

# for regiao in dias_regioes.keys():
#     if regiao in N:
#         dias_regionalizados['N'].append(dias_regioes[regiao])
#     elif regiao in NE:
#         dias_regionalizados['NE'].append(dias_regioes[regiao])
#     elif regiao in SE:
#         dias_regionalizados['SE'].append(dias_regioes[regiao])
#     elif regiao in S:
#         dias_regionalizados['S'].append(dias_regioes[regiao])
#     else:
#         dias_regionalizados['OUTRO'].append(regiao)
