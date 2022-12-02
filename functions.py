def run_fts(train, test, model_name):
  from pyFTS.partitioners import Grid, Util as pUtil
  from pyFTS.models.multivariate import common, variable, mvfts
  from pyFTS.models.seasonal import partitioner as seasonal
  from pyFTS.models.seasonal.common import DateTime

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
