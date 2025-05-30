
from scipy import signal
import xarray as xr

def passa_banda():
    # Filtragem passa-banda entre 3 e 30 dias
    # OTIMIZADO PARA DADOS DIARIOS
    dt = 1  # Frequência de amostragem (1 dia)
    pesos = 4
    # else:
    #     dt = 1
    #     pesos = 15
    periodo_min = 3  # Período mínimo em dias
    periodo_max = 30  # Período máximo em dias
    cutoff_max = 1 / periodo_min
    cutoff_min = 1 / periodo_max
    fn = 1/(2*dt) # frequencia de Nyquist
    b, a = signal.butter(pesos, [cutoff_min/fn, cutoff_max/fn],
                        btype='bandpass')  # Filtro passa-banda de ordem 2

    return (b,a)


def filtra_reanalise(model):
    b, a = passa_banda()
    filtered_model = signal.filtfilt(b, a, model['ssh'], axis=0)

    filtered_ds = xr.DataArray(
                filtered_model,
                coords=model['ssh'].coords,
                dims=model['ssh'].dims,
                attrs=model['ssh'].attrs
                )

    return filtered_ds


def filtra_reanalise_u(model):
    b, a = passa_banda()
    filtered_model = signal.filtfilt(b, a, model['u'], axis=0)

    filtered_ds = xr.DataArray(
                filtered_model,
                coords=model['u'].coords,
                dims=model['u'].dims,
                attrs=model['u'].attrs
                )

    return filtered_ds


def filtra_reanalise_v(model):
    b, a = passa_banda()
    filtered_model = signal.filtfilt(b, a, model['v'], axis=0)

    filtered_ds = xr.DataArray(
                filtered_model,
                coords=model['v'].coords,
                dims=model['v'].dims,
                attrs=model['v'].attrs
                )

    return filtered_ds


def filtra_reanalise_along(model):
    b, a = passa_banda()
    filtered_model = signal.filtfilt(b, a, model['along_shore'], axis=0)

    filtered_ds = xr.DataArray(
                filtered_model,
                coords=model['along_shore'].coords,
                dims=model['along_shore'].dims,
                attrs=model['along_shore'].attrs
                )

    return filtered_ds