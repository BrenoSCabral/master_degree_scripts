'''
Script criado para fazer as analises de wavelet.
'''
import waipy
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from scipy import signal

from read_reanalisys import set_reanalisys_dims
import model_filt


def perform_wavelet(latlon, dt=1, pad=1, dj=.25, s0=2, j1=64, lag1=None):
    model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')

    # model_series = reanal2.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')

    filtered_reanal = model_filt.filtra_reanalise(model_series)

    filt_model=filtered_reanal.values*100

    model_data = model_series['ssh'].values * 100

    data_norm = waipy.normalize(model_data)

    if lag1==None:
        lag1 =  np.corrcoef(data_norm[0:-1], data_norm[1:])[0,1] # essa e a variavel que mais impacta na coerencia do sinal.


    param = 6
    mother = 'Morlet'
    result = waipy.cwt(data=data_norm, 
                    dt=dt, 
                    pad=pad,
                    dj=dj,
                    s0=s0,
                    j1=j1,
                    lag1=lag1,
                    param=param,
                    name='BRAN',
                    mother=mother)

    return result, model_series, filt_model, data_norm, model_data


def plot_wavelet_year(model_series, data_norm, dtmin, result, filt_model, latlon, fig_path):
    time = np.asarray(model_series.time.time)

    # Plot da ondaleta pra cada ano:
    anos = range(1993, 2023)
    for ano in anos:
        label = ''

        t0 = np.datetime64(f'{ano}-01-01')
        tf = np.datetime64(f'{ano}-12-31')

        ticks = []

        for i in range(3,31,3):
            ticks.append(np.log2(i))

        cbar_levels = [.00001, .1, 1, 8, 16, 32, 64, 128, 512, 1024, 2048]

        ########## IMPORTANTE
        # Eh necessario incluir as variaveis nos nomes
        # incluir variavel de energia, variavel de periodo...
        ####################


        waipy.wavelet_plot(label, time, data_norm, dtmin, result,  custom_data=filt_model, ylabel_data='Filtered SSH (cm)', extra_plot=[False],             
                   xmin=t0, xmax=tf, ymin=np.log2(3), ymax=np.log2(30), spectrum_max=10, yticks=ticks, cmap_levels=cbar_levels)
        # plt.savefig(f'/Users/breno/mestrado/wavelet_intervals_test/{latlon[0]}/{ano}.png')
        # plt.show()
        plt.savefig(fig_path + f'/{ano}.png')
        # plt.savefig('/Users/breno/mestrado/wavelet_energy/cbar')
        plt.close('all')


def plot_wavelet_period(model_series, data_norm, dtmin, result, filt_model, latlon, fig_path):
    time = np.asarray(model_series.time.time)

    label = ''

    t0 = np.datetime64(f'1993-01-01')
    tf = np.datetime64(f'2003-12-31')

    ticks = []

    for i in range(3,31,3):
        ticks.append(np.log2(i))

    cbar_levels = [.00001, .1, 1, 8, 16, 32, 64, 128, 512, 1024, 2048]

    ########## IMPORTANTE
    # Eh necessario incluir as variaveis nos nomes
    # incluir variavel de energia, variavel de periodo...
    ####################


    waipy.wavelet_plot(label, time, data_norm, dtmin, result,  custom_data=filt_model, ylabel_data='Filtered SSH (cm)', extra_plot=[False],             
                xmin=t0, xmax=tf, ymin=np.log2(3), ymax=np.log2(30), spectrum_max=10, yticks=ticks, cmap_levels=cbar_levels)
    # plt.savefig(f'/Users/breno/mestrado/wavelet_intervals_test/{latlon[0]}/{ano}.png')
    # plt.show()
    plt.savefig(fig_path + f'/10_anos.png')
    # aqui antes estava periodo_total com o t0 em 010193 e o tf em 311223
    # plt.savefig('/Users/breno/mestrado/wavelet_energy/cbar')
    plt.close('all')



# Passo 3: Selecionar faixa de frequências
def integra_energia(faixa_min, faixa_max):
    mask = (periods >= faixa_min) & (periods < faixa_max)
    energia_selecionada = wavelet_power[mask, :]

    # Passo 4: Integrar a energia ao longo da faixa selecionada
    energia_integrada = np.sum(energia_selecionada, axis=0)
    return energia_integrada


def integra_energia_div(faixa_min, faixa_max):
    mask = (periods >= faixa_min) & (periods < faixa_max)
    energia_selecionada = wavelet_power[mask, :]

    # Passo 4: Integrar a energia ao longo da faixa selecionada
    energia_integrada = np.sum(energia_selecionada, axis=0)/mask.sum()
    return energia_integrada


def integra_energia_alt(faixa_min, faixa_max):
    mask = (periods >= faixa_min) & (periods < faixa_max)
    energia_selecionada = wavelet_power[mask, :]

    # Passo 4: Integrar a energia ao longo da faixa selecionada
    energia_integrada = np.sum(energia_selecionada, axis=0)/(faixa_max - faixa_min)
    return energia_integrada



def plot_year_energies(df_energies, df_filt, mod):

    for year in years:

        df_filt_p = df_filt.loc[str(year)]
        df_energies_p = df_energies[str(year)]

        fig, axes = plt.subplots(2, 1, figsize=(18, 10))
        axes[0].plot(df_filt_p)

        for col in df_energies_p.columns:
            axes[1].plot(df_energies_p[col], label = col)
        
        plt.legend()

        os.makedirs('/Users/breno/mestrado/wavelet_energy/' + mod, exist_ok=True)

        plt.savefig('/Users/breno/mestrado/wavelet_energy/' + mod + f'{year}.png')

        plt.close('all')


def plot_bands(stats, faixas):

    colors = ['#000000', '#595959', '#a0a0a0']

    plt.figure(figsize=(12, 6))

    for i, faixa in enumerate(faixas):
        # Médias e desvios padrão para cada faixa
        num_faixas = len(faixas)  
        offset = (i - num_faixas / 2 + 0.5) * 0.2  # Ajuste o 0.2 conforme o espaçamento desejado

        means = stats[faixa, 'mean']
        stds = stats[faixa, 'std']

        if faixa =='Energia total':
            lbl = '3 to 30 days'
            # Plotar as barras
            plt.bar(
                means.index + offset,  # Deslocamento para as barras
                means.values, 
                yerr=stds.values,  # Barras de erro (desvio padrão)
                width=0.2, 
                label=lbl, 
                color='#a0a0a0',
                error_kw=dict(ecolor='#000000'),
                alpha=0.7,
                capsize=5
            )
        else:
            lbl = f"{faixa} days"
            # Plotar as barras
            plt.bar(
                means.index + offset,  # Deslocamento para as barras
                means.values, 
                yerr=stds.values,  # Barras de erro (desvio padrão)
                width=0.2, 
                label=lbl, 
                color=colors[i],
                error_kw=dict(ecolor=colors[i+1]),
                alpha=0.7,
                capsize=5
            )

    # Ajustes nos eixos e rótulos
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # plt.xlabel("Month")
    # plt.ylabel("Energia Integrada Média [cm²]")
    plt.ylabel('Energy (cm²/cpd)')
    # plt.title("Energia Média por Mês")
    plt.legend(title="Frequency Band")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Salvar ou mostrar o gráfico
    plt.tight_layout()
    plt.savefig(f'/Users/breno/mestrado/wavelet_energy/{latlon[0]}/energia_mensal_band.png', dpi=300)
    plt.close('all')
    # plt.show()


def plot_single_bands(stats, ylbl, img_name):

    plt.figure(figsize=(12, 6))

    means = stats['mean']
    stds = stats['std']
    
    # Plotar as barras
    plt.bar(
        means.index,  # Deslocamento para as barras
        means.values, 
        yerr=stds.values,  # Barras de erro (desvio padrão)
        width=0.2,
        color='black',
        error_kw=dict(ecolor='gray'),
        alpha=0.7,
        capsize=5
    )

    # Ajustes nos eixos e rótulos
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # plt.xlabel("Month")
    plt.ylabel(ylbl)
    # plt.title("Energia Média por Mês")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Salvar ou mostrar o gráfico
    plt.tight_layout()
    plt.savefig(f'/Users/breno/mestrado/wavelet_energy/{latlon[0]}/{img_name}.png', dpi=300)
    plt.close('all')
    # plt.show()


def plot_double_bands(stats1, stats2, ylbl1, ylbl2, img_name):

    fig, ax1 = plt.subplots(figsize=(12, 6))

    means1 = stats1['mean']
    stds1 = stats1['std']

    means2 = stats2['mean']
    stds2 = stats2['std']

    color = 'green'
    ax1.set_ylabel(ylbl1, color=color)
    ax1.bar(means1.index - .1, means1.values, width=.2, yerr=stds1.values, bottom=0,  error_kw=dict(ecolor='lightgrey'), alpha=.7,  color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'red'
    ax2.set_ylabel(ylbl2, color=color)  # we already handled the x-label with ax1
    ax2.bar(means2.index + .1, means2.values, width=.2, yerr=stds2.values, error_kw=dict(ecolor='lightgrey'), alpha=.7, color=color, bottom=0)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig(f'/Users/breno/mestrado/wavelet_energy/{latlon[0]}/{img_name}.png', dpi=300)
    plt.close('all')
    


def count_waves_by_month(df, ssh_col='ssh'):
    """
    Conta o número de ondas (passagens por zero) em uma série temporal e agrupa por mês.

    Parâmetros:
        df (pd.DataFrame): DataFrame com índice datetime e uma coluna ssh.
        ssh_col (str): Nome da coluna com os valores de ssh.

    Retorna:
        pd.DataFrame: Número de ondas registradas por mês.
    """
    # 1. Identificar passagens por zero
    ssh = df[ssh_col].values
    zero_crossings = np.where(np.diff(np.sign(ssh)) != 0)[0]  # Índices onde ocorre a passagem por zero
    
    # 2. Contar meias ondas
    num_half_waves = len(zero_crossings)
    num_full_waves = num_half_waves // 2  # Uma onda completa = 2 meias ondas
    
    # 3. Associar as passagens por zero com as datas
    crossing_dates = df.index[zero_crossings]
    crossing_df = pd.DataFrame({'crossing_date': crossing_dates, 'half_wave': 1})
    
    # 4. Agrupar as contagens por mês
    crossing_df['month'] = crossing_df['crossing_date'].dt.to_period('M')  # Agrupa por mês
    monthly_wave_counts = crossing_df.groupby('month')['half_wave'].sum() // 2  # Contar ondas completas
    
    # 5. Criar DataFrame final
    monthly_wave_counts = monthly_wave_counts.reset_index()
    monthly_wave_counts.columns = ['Month', 'Wave_Count']
    
    return monthly_wave_counts



def count_waves_by_year(df, ssh_col='ssh'):
    """
    Conta o número de ondas (passagens por zero) em uma série temporal e agrupa por ano.

    Parâmetros:
        df (pd.DataFrame): DataFrame com índice datetime e uma coluna ssh.
        ssh_col (str): Nome da coluna com os valores de ssh.

    Retorna:
        pd.DataFrame: Número de ondas registradas por ano.
    """
    # 1. Identificar passagens por zero
    ssh = df[ssh_col].values
    zero_crossings = np.where(np.diff(np.sign(ssh)) != 0)[0]  # Índices onde ocorre a passagem por zero
    
    # 2. Associar as passagens por zero com as datas
    crossing_dates = df.index[zero_crossings]
    crossing_df = pd.DataFrame({'crossing_date': crossing_dates, 'half_wave': 1})
    
    # 3. Agrupar as contagens por ano
    crossing_df['year'] = crossing_df['crossing_date'].dt.year  # Extrair o ano
    yearly_wave_counts = crossing_df.groupby('year')['half_wave'].sum() // 2  # Contar ondas completas
    
    # 4. Criar DataFrame final
    yearly_wave_counts = yearly_wave_counts.reset_index()
    yearly_wave_counts.columns = ['year', 'Wave_Count']
    
    return yearly_wave_counts



def plot_year(df, name, col, ylbl):
    plt.figure(figsize=(12, 6))

    for lat in df.keys():
        plt.plot(df[lat].groupby('year').sum()[col], marker='.', label = lat)


    # Configurações do gráfico
    plt.xlabel('Year')
    plt.ylabel(ylbl) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(df[lat].groupby('year').sum().index, rotation=45)  # Mostrar todos os anos no eixo X
    plt.legend(title='Latitude')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'/Users/breno/mestrado/wavelet_energy/{name}_anual.png', dpi=300)


# testando os slopes
# from scipy import stats
# for lat in df.keys():
#     slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,10957),  np.asarray(df[lat][col]))
#     print(slope)
#     print(f"p-value: {p_value:.4f}")


def plot_year_m(df, name, col, ylbl):
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'orange', 'red', 'green']
    c = 0

    for lat in df.keys():
        plt.plot(df[lat].groupby('year').mean()[col], marker='.', label = lat, color=colors[c])
        # z = np.polyfit(df[lat].groupby('year').mean().index, df[lat].groupby('year').mean()[col], 1)
        # p = np.poly1d(z)
        # m = z[0]
        # b=z[1]
        # plt.plot(df[lat].groupby('year').mean().index, p(df[lat].groupby('year').mean().index), '--', color=colors[c], alpha=0.5)
        c+=1

    # Configurações do gráfico
    plt.xlabel('Year')
    plt.ylabel(ylbl) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(df[lat].groupby('year').mean().index, rotation=45)  # Mostrar todos os anos no eixo X
    plt.legend(title='Latitude')


    plt.tight_layout()
    # plt.show()
    plt.savefig(f'/Users/breno/mestrado/wavelet_energy/pure_mean_{name}_anual.png', dpi=300)



# definir os pontos
pts =[
    (-30, -49.675),
    (-20, -39.862634),
    (-15, -38.843861),
    (-10, -35.753296)
]


# importar e filtrar os dados (tudo do BRAN)
model = 'BRAN'
reanal = {}
years = range(1993, 2023)
for year in years:
    # reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Volumes/BRENO_HD/BRAN/' + str(year)  + '/*.nc')
    #                                     , model)
    reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Users/breno/mestrado/REANALISES_TEMP/BRAN/' + str(year)  + '/*.nc')
                                        , model)

reanalisys = xr.concat(list(reanal.values()), dim="time")


results = {}
energies = {}
m_waves = {}
filts = {}

for latlon in pts:
    fig_path = f'/Users/breno/mestrado/wavelet_energy/{latlon[0]}'
    os.makedirs(fig_path, exist_ok=True)

    print(latlon)

    result, model_series, filt_model, data_norm, model_data = perform_wavelet(latlon, dj= .5)

    results[latlon[0]] = result

    # f, psd = signal.welch(data_norm, fs=1, nperseg=365)
    # f2, psd2 = signal.welch(model_data, fs=1, nperseg=365)
    # f3, psd3 = signal.welch(filt_model, fs=1, nperseg=365)

    # plt.plot(1/f[12:], psd[12:] * np.var(model_data) + 50)
    # plt.plot(1/f2[12:], psd2[12:], color='red')

    # plt.show()

    dtmin = 1/1024

    result['power'] = result['power'] * np.var(model_data)

    plot_wavelet_year(model_series, data_norm, dtmin, result, filt_model, latlon, fig_path)

    plot_wavelet_period(model_series, data_norm, dtmin, result, filt_model, latlon, fig_path)



    wavelet_power = result['power']  # Potência wavelet
    periods = result['period']  # Períodos correspondentes

    time = np.asarray(model_series.time.time)


    df_filt = pd.DataFrame(index=time)
    df_filt['ssh'] = filt_model

    intervals = [3, 16, 30]

    df_energy = pd.DataFrame(index=time)

    df_energy['Energia total'] = integra_energia(3, 30)



    for inter in range(len(intervals)):
        if inter ==0:
            continue
        df_energy[f'{intervals[inter-1]} to {intervals[inter]}'] = integra_energia(intervals[inter-1], intervals[inter])


    plot_year_energies(df_energy, df_filt, f'/{latlon[0]}/energy_band_')
    df_energy.index = pd.to_datetime(df_energy.index)


    # pegar distribuicao por mes

    # Adicionar uma coluna com o número do mês (1 = Janeiro, 2 = Fevereiro, ...)
    df_energy['month'] = df_energy.index.month
    df_energy['year'] = df_energy.index.year

    # Calcular a média e desvio padrão de energia por faixa para cada mês
    stats = df_energy.groupby('month').agg(['mean', 'std'])


    # Configuração para o gráfico
    faixas = [f'3 to 16', f'16 to 30', 'Energia total']

    plot_bands(stats, faixas)



    # plotar quantidade de ondas por mes + desvpad

    month_wave = count_waves_by_month(df_filt)
    month_wave.index = month_wave['Month']
    month_wave = month_wave.drop('Month', axis=1)
    month_wave['month'] = month_wave.index.month
    month_wave['year'] = month_wave.index.year

    stats_mw = month_wave.groupby('month').agg(['max', 'min', 'mean', 'std'])
    plot_single_bands(stats_mw['Wave_Count'], 'Quantity of waves', 'qtd_wave')


    # plotar altura media de onda por mes + desvpad

    df_filt['month'] = df_filt.index.month
    df_filt['year'] = df_filt.index.year

    stats_wh = (np.abs(df_filt)).groupby('month').agg(['max', 'min', 'mean', 'std'])
    plot_single_bands(stats_wh['ssh'], 'Mean wave amplitude [cm]', 'wave_amplitude')


    plot_double_bands(stats_mw['Wave_Count'], stats_wh['ssh'], 'Quantity of Waves', 'Mean Wave Height (cm)', 'double')

    energy = np.asarray(stats['Energia total']['mean'])
    waves = np.asarray(stats_mw['Wave_Count']['mean'])
    height = np.asarray(stats_wh['ssh']['mean'])

    s = ''
    s +=f' MEDIA MENSAL {latlon[0]}:\n'
    s+=f'Energia x Altura : {np.round(np.corrcoef(energy, height)[0,1], 2)}\n'
    s+=f'Energia x Qtd : {np.round(np.corrcoef(energy, waves)[0,1], 2)}\n'
    s+=f'Altura x Qtd : {np.round(np.corrcoef(height, waves)[0,1], 2)}\n'
    text_file = open(f'/Users/breno/mestrado/wavelet_energy/stats_{latlon[0]}.txt', "w")
    text_file.write(s)
    text_file.close()

    print(f' MEDIA MENSAL {latlon[0]}:')

    print(f'Energia x Altura : {np.round(np.corrcoef(energy, height)[0,1], 2)}' )
    print(f'Energia x Qtd : {np.round(np.corrcoef(energy, waves)[0,1], 2)}' )
    print(f'Altura x Qtd : {np.round(np.corrcoef(height, waves)[0,1], 2)}' )

    m_waves[latlon[0]] =  count_waves_by_year(df_filt)
    energies[latlon[0]] = df_energy
    filts[latlon[0]] = np.abs(df_filt)


plot_year(energies, 'energy', 'Energia total', 'Integrated Energy [cm²]')
plot_year(m_waves, 'n_waves', 'Wave_Count', 'Quantity of waves')
plot_year(filts, 'heights', 'ssh', 'Mean Wave Height [cm]')

plot_year_m(energies, 'energy', 'Energia total', 'Integrated Energy [cm²]')
plot_year_m(m_waves, 'n_waves', 'Wave_Count', 'Quantity of waves')
plot_year_m(filts, 'heights', 'ssh', 'Mean Wave Height [cm]')



for lat in energies.keys():
    print(lat)
    print( energies[lat]['Energia total'].std())
    # print( energies[lat]['Energia total'].var())
    print( energies[lat]['Energia total'].mean())
    print((energies[lat]['Energia total'].std()/energies[lat]['Energia total'].mean()) *100)
    print(' ')

    e_y = energies[lat].groupby('year').mean()['Energia total']
    w_y = m_waves[lat].groupby('year').mean()['Wave_Count']
    f_y = filts[lat].groupby('year').mean()['ssh']

    print(f' MEDIA {lat}:')
    print(f'Energia x Altura : {np.round(np.corrcoef(e_y, f_y)[0,1], 2)}' )
    print(f'Energia x Qtd : {np.round(np.corrcoef(e_y, w_y)[0,1], 2)}' )
    print(f'Altura x Qtd : {np.round(np.corrcoef(f_y, w_y)[0,1], 2)}' )




for lat in energies.keys():
    e_y = energies[lat].groupby('year').sum()['Energia total']
    w_y = m_waves[lat].groupby('year').sum()['Wave_Count']
    f_y = filts[lat].groupby('year').sum()['ssh']

    print(f' SOMA {lat}:')
    print(f'Energia x Altura : {np.round(np.corrcoef(e_y, f_y)[0,1], 2)}' )
    print(f'Energia x Qtd : {np.round(np.corrcoef(e_y, w_y)[0,1], 2)}' )
    print(f'Altura x Qtd : {np.round(np.corrcoef(f_y, w_y)[0,1], 2)}' )

plt.close('all')


