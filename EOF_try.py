import xarray as xr
from read_reanalisys import set_reanalisys_dims


model = 'BRAN'
reanal = {}
years = range(1993, 1995)
for year in years:
    # reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Volumes/BRENO_HD/BRAN/' + str(year)  + '/*.nc')
    #                                     , model)
    reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Users/breno/mestrado/REANALISES_TEMP/BRAN/' + str(year)  + '/*.nc')
                                        , model)

reanalisys = xr.concat(list(reanal.values()), dim="time")



data = reanalisys['ssh']
data_centered = data - data.mean(dim='time')

from eofs.xarray import Eof

# Criar o solver EOF
solver = Eof(data_centered)

# Calcular as EOFs (padrões espaciais)
eofs = solver.eofs(neofs=3)  # Primeiras 3 EOFs

# Componentes Principais (PCs) - padrões temporais
pcs = solver.pcs(npcs=3)

# Variância explicada por cada modo
variance = solver.varianceFraction()

# Criar o solver EOF
solver = Eof(data_centered)

# Calcular as EOFs (padrões espaciais)
eofs = solver.eofs(neofs=3)  # Primeiras 3 EOFs

# Componentes Principais (PCs) - padrões temporais
pcs = solver.pcs(npcs=3)

# Variância explicada por cada modo
variance = solver.varianceFraction()



############
