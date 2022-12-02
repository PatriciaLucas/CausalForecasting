def predictArima(series_tr, series_val, n_previsoes, order, seasonal_order, model_fit):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import warnings
    import numpy as np
    import pandas as pd
    import random
    import math
    rmse= []
    yhat = np.zeros((series_val.shape[0],n_previsoes))
    # load and prepare datasets
    dataset = series_tr
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = series_val
    y = validation.values.astype('float32')
    
    # make first prediction
    predictions = list()
    yhat[0,:] = model_fit.forecast(steps=n_previsoes)[0]
    predictions.append(yhat)
    history.append(y[0])
    for i in range(1, len(y)):
      model = sm.tsa.statespace.SARIMAX(history,order=order,seasonal_order=seasonal_order)
      model_fit = model.fit(disp=0)
      yhat[i,:] = model_fit.forecast(steps=n_previsoes)[0]
      predictions.append(yhat)
      obs = y[i]
      history.append(obs)

    for i in range(n_previsoes):
            rmse.append(np.sqrt(mean_squared_error(yhat[0:(yhat.shape[0])-i,i],y[i:])))
    return rmse, yhat, y

def run_arima(train, test):
  from statsmodels.tsa.arima_model import ARIMA
  
  order = (2,2,0)
  seasonal_order = (0,1,0,12)
  model_AR = ARIMA(train.values, order=order)
  model_fit = model_AR.fit(disp=0)
  rmse, yhat, y_test = predictArima(train, test, 1,  order, seasonal_order, model_fit)
  return rmse, y_test, yhat


def run_fts(train, test, model_name):
  from pyFTS.partitioners import Grid, Util as pUtil
  from pyFTS.models.multivariate import common, variable, mvfts
  from pyFTS.models.seasonal import partitioner as seasonal
  from pyFTS.models.seasonal.common import DateTime
  from pyFTS.common import Util
  from pyFTS.benchmarks import Measures
  from pyFTS.partitioners import Grid
  from pyFTS.models import hofts, pwfts
  from pyFTS.common import Membership
  from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid
  from pyFTS.models.multivariate import common, variable, mvfts

  #Definição das variáveis
  UH = variable.Variable("UR", data_label="UR", alias='UR', partitioner=Grid.GridPartitioner, npart=16, data=train) 
  T_MAX = variable.Variable("T_MAX", data_label="T_MAX", alias='T_MAX', partitioner=Grid.GridPartitioner, npart=16, data=train) 
  T_MIN = variable.Variable("T_MIN", data_label="T_MIN", alias='T_MIN', partitioner=Grid.GridPartitioner, npart=16, data=train) 
  V = variable.Variable("V", data_label="V", alias='V', partitioner=Grid.GridPartitioner, npart=16, data=train) 
  R = variable.Variable("R", data_label="R", alias='R', partitioner=Grid.GridPartitioner, npart=16, data=train) 
  ETO = variable.Variable("ETO", data_label="ETO", alias='ETO', partitioner=Grid.GridPartitioner, npart=16, data=train) 

  #Treino e teste dos modelos WeightedMVFTS e MVFTS
  from ssl import VERIFY_CRL_CHECK_CHAIN
  from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid
  import matplotlib.pyplot as plt
  from pyFTS.partitioners import Grid

  models = []
  parameters = [{},{},{'order':2, 'knn': 1},]

  for ct, method in enumerate([model_name]):
    try:
      model = method(explanatory_variables=[UH, T_MAX, T_MIN, V, R], target_variable=ETO, **parameters[ct])
      model.shortname += str(ct)
      model.fit(train)
      models.append(model.shortname)
      forecasts = model.predict(test)
      Util.persist_obj(model, model.shortname)
      del(model)
    except Exception as ex:
      print(method, parameters[ct])
      print(ex)
  
  #Calcula RMSE
  from pyFTS.benchmarks import Measures
  import pandas as pd
  rows = []
  for file in models:
    try:
      model = Util.load_obj(file)
      row = [model.shortname, model.order,len(model)]
      if model.is_multivariate:
        rmse,_,_ = Measures.get_point_statistics(test, model)
        row.append(rmse)
      else:
        rmse,_,_ = Measures.get_point_statistics(test, model)
        row.append(rmse)
      rows.append(row)
    except:
      pass
  results = pd.DataFrame(rows,columns=["Model","Order","Size","RMSE"]).sort_values(["RMSE","Size"])
  return results['RMSE'], test, forecasts
