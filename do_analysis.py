# import custom libraries
import le_dado
import le_reanalise
import filtro
import stats

# import genreal use libraries

# defining paths

image_path = '/Users/breno/Documents/Mestrado/tese/figs/rep3/'
ranalise_path = '/Users/breno/Documents/Mestrado/tese/dados/reanalise/'
data_file = 'Ilha Fiscal 2014.txt'
data_name = 'Ilha Fiscal'


# import and treat data - already apply the filter
data_path = '/Users/breno/dados_mestrado/dados/'
formato = 'csv'
data = le_dado.roda_analise(data_file, data_path, formato=formato, nome=data_name,
                            metodo='composto', fig_folder=image_path)

# importing reanalysis

glor_path = 'GLOR4/'