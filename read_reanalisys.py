def set_reanalisys_dims(reanalisys, name):
    if name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': 'latitude', 'lon': 'longitude', 'surf_el':'ssh'})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'yt_ocean': 'latitude', 'xt_ocean': 'longitude', 'Time':'time', 'eta_t':'ssh'})
    else:
    	for i in list(reanalisys.variables):
            default = ['latitude', 'longitude', 'time']
            if i not in default:
                reanalisys = reanalisys.rename({i:'ssh'})

    return reanalisys


def get_lat_lon(reanalisys):
    return (reanalisys.latitude.values, reanalisys.longitude.values)
