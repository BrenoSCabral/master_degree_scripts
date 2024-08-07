import cdsapi # pra futuros usos vai ser preciso mudar a versao disso
import os

'''
PARA MELHOR ENTENDIMENTO DESTE SCRIPT, CHECAR A PAGINA DO CDS
'''

c = cdsapi.Client()

for i in range(1980, 2023):
    year = str(i)

    print('fazendo o pedido para o ano ' + year)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': '10m_v_component_of_wind',
            'year': [
                year,
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        'area': [
            0, -75, -62,
            -20,
        ],
        'format': 'netcdf',
    },
    f'/Users/breno/mestrado/CPAM/models/ERA5/era5_v_{year}.nc')

    # os.system('mv era5_'+year+'.nc' +' /era5' )