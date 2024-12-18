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

from read_reanalisys import set_reanalisys_dims
import model_filt

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
    reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Volumes/BRENO_HD/BRAN/' + str(year)  + '/*.nc')
                                        , model)

reanalisys = xr.concat(list(reanal.values()), dim="time")

# latlon = pts[0]

for latlon in pts:
    print(latlon[0])
    model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')
    filtered_reanal = model_filt.filtra_reanalise(model_series)

    filt_model=filtered_reanal.values*100

    model_data = model_series['ssh'].values * 100
    # model_nfilt = model_series['ssh'].values

    # model_data_cut =  model_series.sel(time=slice('2000' , '2004'))['ssh'].values # selecionando um ano especifico

    # funcoes da wavelet
    # data_norm = waipy.normalize(model_nfilt)
    # data_norm = waipy.normalize(model_data[:365*2])
    data_norm = waipy.normalize(model_data)

    # TODO -> Ajustar dt e j1, testar diferentes lags
    n1 = len(data_norm)
    dt =  1 # dado diario -> botar 1
    pad = 1 # bota 0 no inicio e final da serie
    dj = .5 # 0.25 isto faz 4 sub-oitavas por oitava -> como performa a wavelet
    s0 = 2*dt # escala inicial do dominio de periodo
    j1 =  7/dj # int(np.floor((np.log10(n1*dt/s0))/np.log10(2)/dj))   # isso diz fazer 7 potências de dois com dj sub-oitavas cada -> ate onde vai a ondaleta
                                                                    # no meu caso essa variavel nao ta importando tanto

    lag1 =  np.corrcoef(data_norm[0:-1], data_norm[1:])[0,1] # essa e a variavel que mais impacta na coerencia do sinal.
    param = 6
    mother = 'Morlet'
    dtmin = 1/256# 0.25/8
    # data_norm = waipy.normalize(model)
    # alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0,1]; 
    # print("Lag-1 autocorrelation = {:4.2f}".format(alpha))


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
    time = np.asarray(model_series.time.time)
    cmap_levels = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] # , 64.0# , 128.0, 256.0]

    # Plot da ondaleta pra cada ano:
    anos = range(1993, 2024)
    for ano in anos:
        label = 'BRAN'

        # aqui embaixo ta plotando a serie filtrada em cima
        t0 = np.datetime64(f'{ano}-01-01')
        tf = np.datetime64(f'{ano}-12-31')

        ticks = []

        for i in range(3,31,3):
            ticks.append(np.log2(i))


        waipy.wavelet_plot(label, time, data_norm, dtmin, result,  custom_data=filt_model, ylabel_data='ASM Filtrada (cm)', extra_plot=[False],             
                    xmin=t0, xmax=tf, ymin=np.log2(3), ymax=np.log2(30), cmap_levels=cmap_levels, spectrum_max=10, yticks=ticks,)
        plt.savefig(f'/Users/breno/mestrado/ondaleta/{latlon[0]}/{ano}.png')
        plt.close('all')

    # # isso aqui faz o plot com a energia integrada

    # wavelet_power = result['power']  # Potência wavelet
    # periods = result['period']  # Períodos correspondentes

    # # Passo 3: Selecionar faixa de frequências
    # faixa_min, faixa_max = 8, 16
    # mask = (periods >= faixa_min) & (periods <= faixa_max)
    # energia_selecionada = wavelet_power[mask, :]

    # # Passo 4: Integrar a energia ao longo da faixa selecionada
    # energia_integrada = np.sum(energia_selecionada, axis=0)

    # # Determinar eventos energéticos (por exemplo, quando energia supera um limiar)
    # # limiar = np.mean(energia_total) + 2 * np.std(energia_total)
    # # eventos = energia_total > limiar
    # # numero_eventos = np.sum(eventos)
    # waipy.wavelet_plot(label, time, data_norm, dtmin, result,  custom_data=filt_model, ylabel_data='ASM Filtrada (cm)',extra_plot=energia_integrada,                 
    #             extra_lbl=f'Energia Integrada entre {faixa_min} e {faixa_max} dias',xmin=t0, xmax=tf, ymin=np.log2(3), ymax=np.log2(30), cmap_levels=cmap_levels, spectrum_max=10, yticks=ticks)
    # plt.savefig('/Users/breno/mestrado/ondaleta/testes/test0.png')
    # plt.close('all')

    # levantamento de eventos energeticos:

    wavelet_power = result['power']  # Potência wavelet
    periods = result['period']  # Períodos correspondentes

    # Passo 3: Selecionar faixa de frequências
    def integra_energia(faixa_min, faixa_max):
        mask = (periods >= faixa_min) & (periods < faixa_max)
        energia_selecionada = wavelet_power[mask, :]

        # Passo 4: Integrar a energia ao longo da faixa selecionada
        energia_integrada = np.sum(energia_selecionada, axis=0)
        return energia_integrada




    df_energy = pd.DataFrame(index=time)

    df_energy['Energia total'] = integra_energia(3, 30)
    df_energy['3 a 5'] = integra_energia(3, 5)
    df_energy['5 a 8'] = integra_energia(5, 8)
    df_energy['8 a 16'] = integra_energia(8, 16)
    df_energy['16 a 30'] = integra_energia(16, 30)


    # Converter o índice do DataFrame para datetime, caso ainda não seja
    df_energy.index = pd.to_datetime(df_energy.index)

    #############################
    # plot das OCCs mais altas #
    ############################

    fmodel_df = pd.DataFrame(index=time)
    fmodel_df['ssh'] = filt_model

    perc99_9 = fmodel_df['ssh'].quantile(0.999)
    high_events = fmodel_df[fmodel_df['ssh'] > perc99_9]

    # 3. Criar uma lista com os períodos de 5 dias antes e 5 dias depois
    event_windows = []
    for event_date in high_events.index:
        start_date = event_date - pd.Timedelta(days=15)
        end_date = event_date + pd.Timedelta(days=15)
        event_window = fmodel_df.loc[start_date:end_date]
        event_window['Event'] = event_date  # Identifica a data central do evento
        event_windows.append(event_window)

    # 4. Concatenar todas as janelas de eventos em um DataFrame
    event_windows_df = pd.concat(event_windows)

    # 5. Salvar os resultados no Excel
    output_path = "event_analysis.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        high_events.to_excel(writer, sheet_name="High Events")
        event_windows_df.to_excel(writer, sheet_name="Event Windows")

    # 6. Plotar a série temporal de cada evento
    output_dir = f"/Users/breno/mestrado/ondaleta/{latlon[0]}/extreme_events/"    
    os.makedirs(output_dir, exist_ok=True)




    df_energy2 = df_energy.copy()
    df_energy2['ssh'] = fmodel_df['ssh']

    for event_date in high_events.index:
        # Filtrar o período de 15 dias antes e depois do evento
        start_date = event_date - pd.Timedelta(days=15)
        end_date = event_date + pd.Timedelta(days=15)
        plot_data = df_energy2.loc[start_date:end_date]

        # Preparar os subplots
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 10), constrained_layout=True, sharex=True)

        # 4. Subplot 1: Série temporal de SSH
        axes[0].plot(plot_data.index, plot_data['ssh'], color='blue', label='SSH')
        axes[0].axhline(perc99_9, color='red', linestyle='--', label='Percentil 99.9')
        axes[0].axvline(event_date, color='black', linestyle='--', label='Evento')
        axes[0].set_ylabel("SSH [cm]")
        axes[0].legend(loc="upper left")
        axes[0].set_title(f"Evento em {event_date.date()}")

        # 5. Subplots 2 a 6: Integração de energia em diferentes faixas
        energy_labels = ['Energia total', '3 a 5', '5 a 8', '8 a 16', '16 a 30']
        colors = ['darkblue', 'darkgreen', 'orange', 'purple', 'brown']

        for i, col in enumerate(energy_labels):
            axes[i + 1].plot(plot_data.index, plot_data[col], color=colors[i], label=col)
            axes[i + 1].axvline(event_date, color='black', linestyle='--')  # Linha vertical no evento
            axes[i + 1].set_ylabel(f"{col} [cm²]")
            axes[i + 1].legend(loc="upper left")

        # Configurar eixo x (datas) e ticks
        axes[-1].set_xlabel("Data")
        for ax in axes:
            ax.grid()

        # Salvar o gráfico
        plt.savefig(f"{output_dir}/event_{event_date.date()}.png")
        plt.close()








    # Plot das series de energia
    fig = plt.figure(figsize=(15,7))
    axe = fig.add_subplot(111)

    # Defina os limites do eixo x e y para todos os subgráficos
    x_limits = (pd.Timestamp('1993-01-01'), pd.Timestamp('2023-01-01'))
    y_limits = (0, 32)

    ax1= fig.add_subplot(511)
    ax2= fig.add_subplot(512)
    ax3= fig.add_subplot(513)
    ax4= fig.add_subplot(514)
    ax5= fig.add_subplot(515)
    axes = [ax1, ax2, ax3, ax4, ax5]

    for i, col in enumerate(df_energy.columns):
        ax = axes[i]
        ax.plot(df_energy[col])
        if i == 0:
            ylbl = 'Total'
        else:
            ylbl = f'{col} dias'
        ax.yaxis.set_label_position("right")
        ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Ticks principais a cada ano
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))  # Ticks menores a cada 6 meses
        
        ax.set_ylabel(ylbl)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        
        # Remover rótulos do eixo x dos gráficos superiores
        if i < len(axes) - 1:
            ax.set_xticklabels([])

    # Configurar o último eixo (axes[4])
    axe.spines['top'].set_color('none')
    axe.spines['bottom'].set_color('none')
    axe.spines['left'].set_color('none')
    axe.spines['right'].set_color('none')
    axe.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    axe.set_ylabel('Energia Integrada [cm²]')
    #

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))  # Ticks principais a cada ano
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(interval=6))  # Ticks menores a cada 6 meses
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato do ano
    plt.gcf().autofmt_xdate()

    axes[-1].set_xlabel('Ano')

    plt.tight_layout()
    plt.savefig(f'/Users/breno/mestrado/ondaleta/{latlon[0]}/energy_series.png', dpi=500)
    plt.close()

    # Adicionar uma coluna com o número do mês (1 = Janeiro, 2 = Fevereiro, ...)
    df_energy['month'] = df_energy.index.month
    df_energy['year'] = df_energy.index.year

    # Calcular a média e desvio padrão de energia por faixa para cada mês
    stats = df_energy.groupby('month').agg(['mean', 'std'])

    # Exibir as estatísticas para conferência
    print(stats)

    # Configuração para o gráfico
    faixas = ['3 a 5', '5 a 8', '8 a 16', '16 a 30']
    colors = ['blue', 'green', 'orange', 'red']

    # Criar o gráfico
    plt.figure(figsize=(12, 6))

    for i, faixa in enumerate(faixas):
        # Médias e desvios padrão para cada faixa
        means = stats[faixa, 'mean']
        stds = stats[faixa, 'std']
        
        # Plotar as barras
        plt.bar(
            means.index + i * 0.2,  # Deslocamento para as barras
            means.values, 
            yerr=stds.values,  # Barras de erro (desvio padrão)
            width=0.2, 
            label=f"{faixa} dias", 
            color=colors[i],
            alpha=0.8,
            capsize=5
        )

    # Ajustes nos eixos e rótulos
    plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    plt.xlabel("Mês")
    plt.ylabel("Energia Integrada Média [cm²]")
    # plt.title("Energia Média por Mês")
    plt.legend(title="Faixa de Frequência")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Salvar ou mostrar o gráfico
    plt.tight_layout()
    plt.savefig(f"/Users/breno/mestrado/ondaleta/{latlon[0]}/energia_media_mensal.png", dpi=300)
    plt.close()
    # plt.show()

    
    energy_year = df_energy.groupby('year').sum()

    # plotar energia total por ano

    plt.figure(figsize=(12, 6))
    plt.bar(energy_year.index, energy_year['Energia total'], color='skyblue', edgecolor='black')

    # Configurações do gráfico
    plt.xlabel('Ano')
    plt.ylabel('Energia integrada de OCCs [m²]')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(energy_year.index, rotation=45)  # Mostrar todos os anos no eixo X

    plt.tight_layout()
    plt.savefig(f"/Users/breno/mestrado/ondaleta/{latlon[0]}/energia_anual.png", dpi=300)
    # fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 10), constrained_layout=True)

    #######################################
    # Contando numero de ondas por mes    #
    #######################################


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

    monthly_wave_counts = count_waves_by_month(fmodel_df)


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
        yearly_wave_counts.columns = ['Year', 'Wave_Count']
        
        return yearly_wave_counts

    # -------------------
    # Aplicar a Função
    # -------------------
    yearly_wave_counts = count_waves_by_year(fmodel_df)

    # -------------------
    # Plotar Gráfico de Barras
    # -------------------
    plt.figure(figsize=(12, 6))
    plt.bar(yearly_wave_counts['Year'], yearly_wave_counts['Wave_Count'], color='skyblue', edgecolor='black')

    # Configurações do gráfico
    plt.xlabel('Ano')
    plt.ylabel('Número de Ondas')
    plt.title('Número de Ondas Registradas por Ano')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(yearly_wave_counts['Year'], rotation=45)  # Mostrar todos os anos no eixo X

    plt.tight_layout()
    plt.savefig(f"/Users/breno/mestrado/ondaleta/{latlon[0]}/ondas_anual.png", dpi=300)


    # Plotar gráfico de barra dos dados anuais

    def count_waves_and_monthly_stats(df, ssh_col='ssh'):
        """
        Conta ondas e calcula a média e desvio padrão do número de ondas por mês do ano.

        Parâmetros:
            df (pd.DataFrame): DataFrame com índice datetime e coluna 'ssh'.
            ssh_col (str): Nome da coluna com os valores de ssh.

        Retorna:
            pd.DataFrame: DataFrame com média e desvio padrão do número de ondas por mês do ano.
        """
        # 1. Identificar passagens por zero
        ssh = df[ssh_col].values
        zero_crossings = np.where(np.diff(np.sign(ssh)) != 0)[0]  # Passagens por zero
        
        # 2. Contar ondas
        crossing_dates = df.index[zero_crossings]  # Datas das passagens
        crossing_df = pd.DataFrame({'date': crossing_dates, 'half_wave': 1})
        crossing_df['month'] = crossing_df['date'].dt.month  # Extrair mês
        crossing_df['year'] = crossing_df['date'].dt.year  # Extrair ano
        
        # Contagem de ondas (2 meias ondas = 1 onda completa)
        monthly_counts = crossing_df.groupby(['year', 'month'])['half_wave'].sum() // 2
        monthly_counts = monthly_counts.reset_index()
        monthly_counts.columns = ['Year', 'Month', 'Wave_Count']
        
        # 3. Calcular média e desvio padrão para cada mês ao longo dos anos
        stats = monthly_counts.groupby('Month')['Wave_Count'].agg(['mean', 'std']).reset_index()
        stats.columns = ['Month', 'Mean_Wave_Count', 'Std_Wave_Count']
        
        return stats



    # Calcular estatísticas mensais
    monthly_wave_stats = count_waves_and_monthly_stats(fmodel_df)

    # 4. Plotar gráfico com médias e desvio padrão
    plt.figure(figsize=(10, 6))
    plt.bar(monthly_wave_stats['Month'], monthly_wave_stats['Mean_Wave_Count'], 
            yerr=monthly_wave_stats['Std_Wave_Count'], 
            capsize=5, color='skyblue', edgecolor='black')

    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.xlabel("Mês")
    plt.ylabel("Número de Ondas")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"/Users/breno/mestrado/ondaleta/{latlon[0]}/ondas_media_mensal.png", dpi=300)




'''
ymin=1.4044532162857952, ymax=5.654453216285795,

Fazer abaixo a analise sazonal, separando por algumas faixas: - acho que a melhor forma de exportar isso aqui seria atraves de tabelas

3 a 8
8 a 16
16 a 30

como posso justificar fisicamente essas faixas? DESAFIO

bandas de Castro 1995:

entre 12 e 9.6 dias

entre 6 e 7 dias

entre 4 e 4.4 dias

Bandas adaptadas:

3 a 5
5 a 8
8 a 16
16 a 30
'''


def conta_eventos():
    '''
    TODO:
    1 - Ver a melhor forma de contabilizar essa variabilidade
    2 - Fazer esses resultados por ano


    ---> Fazendo serie dos eventos de OCCs (contabilizando variabilidade sazonal)
    '''
    wavelet_result = result
    # Extrair os resultados principais
    wavelet_power = wavelet_result['power']  # Potência wavelet
    periods = wavelet_result['period']  # Períodos correspondentes
    # time = wavelet_result['time']  # Tempo
    # time = np.asarray(model_series.sel(time=slice('2000' , '2001')).time.time)

    # Passo 3: Selecionar faixa de frequências
    faixa_min, faixa_max = 3, 5
    mask = (periods >= faixa_min) & (periods <= faixa_max)
    energia_selecionada = wavelet_power[mask, :]

    # Passo 4: Integrar a energia ao longo da faixa selecionada
    energia_total = np.sum(energia_selecionada, axis=0)

    # Determinar eventos energéticos (por exemplo, quando energia supera um limiar)
    limiar = np.mean(energia_total) + 2 * np.std(energia_total)
    eventos = energia_total > limiar
    numero_eventos = np.sum(eventos)

    print(f"Número total de eventos energéticos: {numero_eventos}")

    # Passo 5: Plotar os resultados
    # import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.plot(time, energia_total, label="Energia Integrada")
    ax.axhline(y=limiar, color='r', linestyle='--', label="Limiar")
    # ax.scatter(time[eventos], energia_total[eventos], color='red', label="Eventos")
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Energia Integrada')
    ax.legend()
    ax.set_xlim(t0, tf)
    # ax.title('Eventos Energéticos ao Longo do Tempo')
    plt.show()



def faz_analise(ano_i, ano_f, pt, str_pt):
    model = 'BRAN'
    reanal = {}
    years = range(ano_i, ano_f)
    for year in years:
        reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Volumes/BRENO_HD/BRAN/' + str(year)  + '/*.nc')
                                            , model)
        
    reanalisys = xr.concat(list(reanal.values()), dim="time")
    latlon = pt
    model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')
    filtered_reanal = model_filt.filtra_reanalise(model_series)

    model=filtered_reanal.values*100

    # model_nfilt = model_series['ssh'].values


    # funcoes da wavelet
    # data_norm = waipy.normalize(model_nfilt)
    data_norm = waipy.normalize(model)

    dt = 10.25/8 # dado diario
    pad = 1 # bota 0 no inicio e final da serie
    dj = 0.25 # isto faz 4 sub-oitavas por oitava
    s0 = 2*dt # escala de 12h, ja que dado eh diario
    j1 = 8/dj       # isso diz fazer 8 potências de dois com dj sub-oitavas cada
    lag1 = 0.72 # np.corrcoef(data_norm[0:-1], data_norm[1:])[0,1]
    param = 6
    mother = 'Morlet'
    dtmin = 0.25/8


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

    label = 'BRAN'
    time = np.asarray(filtered_reanal.time)

    def wavelet_plot(var, time, data, dtmin, result, **kwargs):
        y1 = 1.4044532162857952
        y2= 5.654453216285795
        """
        PLOT WAVELET TRANSFORM
        var = title name from data
        time  = vector get in load function
        data  = from normalize function
        dtmin = minimum resolution :1 octave
        result = dict from cwt function

        kwargs:
            no_plot
            filename
            xlabel_cwt
            ylabel_cwt
            ylabel_data
            plot_phase : bool, defaults to False

        """
        # frequency limit
        # print result['period']
        # lim = np.where(result['period'] == result['period'][-1]/2)[0][0]
        #"""Plot time series """

        fig = plt.figure(figsize=(15, 10), dpi=300)

        gs1 = gridspec.GridSpec(4, 3)
        gs1.update(
            left=0.07, right=0.7, wspace=0.5, hspace=0, bottom=0.15, top=0.97
        )

        ax1 = plt.subplot(gs1[0, :])
        ax1 = plt.gca()
        ax1.xaxis.set_visible(False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(gs1[1:4, :])  # , axisbg='#C0C0C0')

        gs2 = gridspec.GridSpec(4, 1)
        gs2.update(
            left=0.7, right=0.98, hspace=0, bottom=0.15, top=0.97
        )
        ax5 = plt.subplot(gs2[1:4, 0], sharey=ax2)
        plt.setp(ax5.get_yticklabels(), visible=False)

        gs3 = gridspec.GridSpec(6, 1)
        gs3.update(
            left=0.77, top=0.97, right=0.98, hspace=0.6, wspace=0.01,
        )
        ax3 = plt.subplot(gs3[0, 0])

        ax1.plot(time, data)
        ax1.axis('tight')
        ax1.set_xlim(time.min(), time.max())
        ax1.set_ylabel(kwargs.get('ylabel_data', 'Amplitude'), fontsize=15)
        ax1.set_title('%s' % var, fontsize=17)
        ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune='lower'))
        ax1.grid(True)
        ax1.xaxis.set_visible(False)
        joint_wavelet = result['joint_wavelet']
        wavelet_x = np.arange(-result['nw'] / 2, result['nw'] / 2)
        ax3.plot(
            wavelet_x,
            joint_wavelet.real,
            'k',
            label='Real part'
        )
        ax3.plot(
            wavelet_x,
            joint_wavelet.imag,
            '--k',
            label='Imag part'
        )
        ax3.plot(
            wavelet_x,
            result['mean_wavelet'],
            'g',
            label='Mean'
        )

        # try to infer the xlims by selecting the limit at 5% of maximum value of
        # real part
        limit_index = np.where(
            np.abs(joint_wavelet.real) > 0.05 * np.max(np.abs(joint_wavelet.real))
        )
        ax3.set_xlim(-wavelet_x[limit_index[0][0]], wavelet_x[limit_index[0][0]])
        # ax3.axis('tight')
        # ax3.set_xlim(-100, 100)
        # ax3.set_ylim(-0.3,0.3)
        # ax3.set_ylim(
        #     [np.min(result['joint_wavelet']),np.max(result['joint_wavelet'])])
        ax3.set_xlabel('Time', fontsize=10)
        ax3.set_ylabel('Amplitude', fontsize=10)
        ax3.set_title(r'$\psi$ (t/s) {0} in time domain'.format(result['mother']))
        # ------------------------------------------------------------------------#
        # ax4.plot(result['ondaleta'],'k')
        # ax4.set_xlabel('Frequency', fontsize=10)
        # ax4.set_ylabel('Amplitude', fontsize=10)
        # ax4.set_title('$\psi^-$  Frequency domain', fontsize=13)
        # ------------------------------------------------------------------------#
        # colorbar location
        position = fig.add_axes([0.07, 0.07, 0.6, 0.01])

        plot_phase = kwargs.get('plot_phase', False)
        if plot_phase:
            phases = np.arctan(
                np.imag(result['wave']),
                np.real(result['wave'])
            )
            # import IPython
            # IPython.embed()
            # exit()
            phase_levels = np.linspace(phases.min(), phases.max(), 10)
            norm = matplotlib.colors.DivergingNorm(vcenter=0)
            pc = ax2.contourf(
                time,
                np.log2(result['period']),
                phases,
                phase_levels,
                cmap=mpl.cm.get_cmap('seismic'),
                norm=norm
            )
            cbar = plt.colorbar(
                pc,
                cax=position,
                orientation='horizontal',
            )
            cbar.set_label('Phase [rad]')

        else:
            #""" Contour plot wavelet power spectrum """
            lev = levels(result, dtmin)
            # import IPython
            # IPython.embed()
            # exit()
            cmap = mpl.cm.get_cmap('viridis')
            cmap.set_over('yellow')
            cmap.set_under('cyan')
            cmap.set_bad('red')
            #ax2.imshow(np.log2(result['power']), cmap='jet', interpolation=None)
            #ax2.set_aspect('auto')
            pc = ax2.contourf(
                time,
                np.log2(result['period']),
                np.log2(result['power']),
                np.log2(lev),
                cmap=cmap,
            )
            # print(time.shape)
            # print(np.log2(result['period']).shape)
            # print(np.log2(result['power']).shape)
            # X, Y = np.meshgrid(time, np.log2(result['period']))
            # ax2.scatter(
            #     X.flat,
            #     Y.flat,
            # )

            # 95% significance contour, levels at -99 (fake) and 1 (95% signif)
            pc2 = ax2.contour(
                time,
                np.log2(result['period']),
                result['sig95'],
                [-99, 1],
                linewidths=2
            )
            pc2
            ax2.plot(time, np.log2(result['coi']), 'k')
            # cone-of-influence , anything "below"is dubious
            ax2.fill_between(
                time,
                np.log2(result['coi']),
                int(np.log2(result['period'][-1]) + 1),
                # color='white',
                alpha=0.6,
                hatch='/'
            )

            def cb_formatter(x, pos):
                # x is in base 2
                linear_number = 2 ** x
                return '{:.1f}'.format(linear_number)

            cbar = plt.colorbar(
                pc, cax=position, orientation='horizontal',
                format=mpl.ticker.FuncFormatter(cb_formatter),
            )
            cbar.set_label('Power')

        yt = range(
            int(np.log2(result['period'][0])),
            int(np.log2(result['period'][-1]) + 1)
        )  # create the vector of periods
        Yticks = [float(math.pow(2, p)) for p in yt]  # make 2^periods
        # Yticks = [int(i) for i in Yticks]
        ax2.set_yticks(yt)
        ax2.set_yticklabels(Yticks)
        ax2.set_ylim(
            ymin= y1, # (np.log2(np.min(result['period']))),
            ymax= y2 # (np.log2(np.max(result['period'])))
        )
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.set_xlabel(kwargs.get('xlabel_cwt', 'Time'), fontsize=12)
        ax2.set_ylabel(kwargs.get('ylabel_cwt', 'Period'), fontsize=12)
        # ax2.axhline(y=10.5, xmin=0, xmax=1, linewidth=2, color='k')
        # ax2.axhline(y=13.3, xmin=0, xmax=1, linewidth=2, color='k')

        # if requested, limit the time range that we show
        xmin = kwargs.get('xmin', None)
        xmax = kwargs.get('xmax', None)
        if xmin is not None or xmax is not None:
            for ax in (ax1, ax2):
                ax.set_xlim(xmin, xmax)

        #Plot global wavelet spectrum
        f, sxx = fft(data)
        ax5.plot(
            sxx, np.log2(1 / f * result['dt']), 'gray', label='Fourier spectrum'
        )
        ax5.plot(
            result['global_ws'], np.log2(result['period']), 'b',
            label='Wavelet spectrum'
        )
        ax5.plot(
            result['global_signif'], np.log2(result['period']), 'r--',
            label='95% confidence spectrum'
        )
        ax5.legend(loc=0)
        ax5.set_xlim(0, 1.25 * np.max(result['global_ws']))
        ax5.set_xlabel('Power', fontsize=10)
        ax5.set_title('Global Wavelet Spectrum', fontsize=12)
        # save fig
        if not kwargs.get('no_plot', False):
            filename = kwargs.get('filename', '{}.png'.format(var))
            fig.savefig(filename, dpi=300)

        ret_dict = {
            'fig': fig,
            'ax_data': ax1,
            'ax_cwt': ax2,
            'ax_wavelet': ax3,
            # 'ax:': ax4,
            'ax_global_spectrum': ax5,
        }
        return ret_dict

    wavelet_plot(label, time, data_norm, dtmin, result);plt.savefig(f'/Users/breno/mestrado/ondaleta/{str_pt}_{ano_i}_{ano_f}.png')

interval = 4
ano_i = 1993
ano_f = 1993 + interval

str_pt=0
for pt in pts:
    ano_i = 1993
    ano_f = 1993 + interval
    str_pt+=1
    print(str_pt)
    while ano_f < 2024:
        print('INICIANDO!')
        print(ano_i)
        print(ano_f)
        faz_analise(ano_i, ano_f, pt, str_pt)

        if ano_f == 2023:
            break
        elif ano_f + interval >= 2023:
            ano_i = ano_f
            ano_f = 2023
        else:
            ano_i += interval; ano_f +=interval


