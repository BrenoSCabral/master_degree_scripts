def set_reanalisys_dims(reanalisys, name, lat_name = 'latitude', lon_name = 'longitude', time_name ='time', ssh_name = 'ssh'):
    if name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': lat_name, 'lon': lon_name, 'surf_el':ssh_name})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'yt_ocean': lat_name, 'xt_ocean': lon_name, 'Time':'time', 'eta_t':ssh_name})
    else:
    	for i in list(reanalisys.variables):
            default = [lat_name, lon_name, 'time']
            if i not in default:
                reanalisys = reanalisys.rename({i:ssh_name})

    return reanalisys


def get_lat_lon(reanalisys):
    return (reanalisys.latitude.values, reanalisys.longitude.values)

def set_reanalisys_curr_dims(reanalisys, name, lat_name = 'latitude', lon_name = 'longitude', time_name ='time',
                                 v_name = 'v', u_name = 'u', depth_name='depth'):
    if name == 'CGLO':
        reanalisys = reanalisys.rename({'uo_cglo': u_name, 'vo_cglo': v_name})
    elif name == 'ORAS':
        reanalisys = reanalisys.rename({'uo_oras': u_name, 'vo_oras': v_name})
    elif name == 'FOAM':
        reanalisys = reanalisys.rename({'uo_foam': u_name, 'vo_foam': v_name})
    elif name == 'GLOR4':
        reanalisys = reanalisys.rename({'uo_glor':u_name, 'vo_glor':v_name})
    elif name == 'GLOR12':
        reanalisys = reanalisys.rename({'uo':u_name, 'vo':v_name})
    elif name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': lat_name, 'lon': lon_name, 'water_u':u_name, 'water_v':v_name})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'st_ocean':depth_name, 'yu_ocean': lat_name, 'xu_ocean': lon_name, 'Time':time_name})


    return reanalisys