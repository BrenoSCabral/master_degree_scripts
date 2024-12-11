'''
Script criado para fazer as analises de wavelet.
'''
import waipy
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

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
latlon = pts[0]
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

label = 'BRAN'
time = np.asarray(model_series.time.time)

# aqui embaixo ta plotando a serie filtrada em cima
t0 = np.datetime64('2001-01-01')
tf = np.datetime64('2005-01-01')

ticks = []

for i in range(3,31,3):
    ticks.append(np.log2(i))


wavelet_power = result['power']  # Potência wavelet
periods = result['period']  # Períodos correspondentes

# Passo 3: Selecionar faixa de frequências
faixa_min, faixa_max = 8, 16
mask = (periods >= faixa_min) & (periods <= faixa_max)
energia_selecionada = wavelet_power[mask, :]

# Passo 4: Integrar a energia ao longo da faixa selecionada
energia_integrada = np.sum(energia_selecionada, axis=0)

# Determinar eventos energéticos (por exemplo, quando energia supera um limiar)
# limiar = np.mean(energia_total) + 2 * np.std(energia_total)
# eventos = energia_total > limiar
# numero_eventos = np.sum(eventos)

cmap_levels = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] # , 64.0# , 128.0, 256.0]
waipy.wavelet_plot(label, time, data_norm, dtmin, result,  custom_data=filt_model, ylabel_data='ASM Filtrada (cm)',extra_plot=energia_integrada,                 
             extra_lbl=f'Energia Integrada entre {faixa_min} e {faixa_max} dias',xmin=t0, xmax=tf, ymin=np.log2(3), ymax=np.log2(30), cmap_levels=cmap_levels, spectrum_max=10, yticks=ticks,)
plt.savefig('/Users/breno/mestrado/ondaleta/testes/test0.png')
plt.close('all')



'''
ymin=1.4044532162857952, ymax=5.654453216285795,

Fazer abaixo a analise sazonal, separando por algumas faixas: - acho que a melhor forma de exportar isso aqui seria atraves de tabelas

3 a 8
8 a 16
16 a 30

como posso justificar fisicamente essas faixas? DESAFIO
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


