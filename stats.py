import numpy as np

def stats(model, data):
    '''Função que faz as estatísticas de comparação entre dados e modelo.

    Args:
        model (numpy.ndarray): array dos outputs do modelo filtrados
        data (numpy.ndarray): array dos dados filtrados

    Returns:
        tuple: Resultados das estatísticas de comparação entre dados e modelo
    '''
    if (data/model).mean() > 50 or (data/model).mean() < -50:
        print('Checar se os dados e o modelo estao na mesma unidade')

    sum_data = np.sum(data)
    sum_model = np.sum(model)

    mean_data = np.mean(data)
    mean_model = np.mean(model)

    corr =round(np.corrcoef(model, data)[0,1] * 100, 2)

    # rmse = np.sqrt(np.sum((model - data)**2) /len(data))
    
    nrmse  =  np.sqrt((((sum_model - sum_data)**2)/(sum_data**2)))

    si_up = ((sum_model - mean_model) - (sum_data - mean_data))**2
    si_down = sum_data**2
    si = si_up/si_down

    # bias = np.mean(data - model) # ver a forma de fazer o nbias

    nbias = (sum_model - sum_data) / sum_data

    return(corr, nbias, nrmse, si)


def dependent_stats(model, data):
    '''Faz as estatísticas dependentes de comparação entre dados e modelo.

    Args:
        model (numpy.ndarray): array dos outputs do modelo filtrados
        data (numpy.ndarray): array dos dados filtrados

    Returns:
        tuple: Resultados das estatísticas de comparação entre dados e modelo
    '''
    if (data/model).mean() > 50 or (data/model).mean() < -50:
        print('Checar se os dados e o modelo estao na mesma unidade')

    sum_data = np.sum(data)
    sum_model = np.sum(model)

    mean_data = np.mean(data)
    mean_model = np.mean(model)

    n = len(data)

    bias = (sum_model - sum_data) / n
    
    rmse  =  np.sqrt((((sum_model - sum_data)**2)/n))

    scrmse = np.sqrt(rmse**2 - bias**2)

    # bias = np.mean(data - model) # ver a forma de fazer o nbias


    return(bias, rmse, scrmse)

def general_stats(series):
    '''Faz as estatísticas gerais de uma série.

    Args:
        series (numpy.ndarray): array da série
    '''

    max = np.max(series)

    min = np.min(series)

    sum = np.sum(series)

    mean = np.mean(series)

    std = np.std(series)

    median = np.median(series)

    return(sum, mean, std, median, max, min)
