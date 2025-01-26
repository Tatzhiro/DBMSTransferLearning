from abc import ABC, abstractmethod
from .utils import drop_unimportant_parameters
import pandas as pd
from pandas import DataFrame
import numpy as np
from .system_configuration import SystemConfiguration

class DistributionDistance(ABC):
  @abstractmethod
  def compute_distance(self, df1: DataFrame, df2: DataFrame, parameters: list, system: SystemConfiguration):
    pass

class BhattacharyyaDistance(DistributionDistance):
  def compute_distance(self, df1: DataFrame, df2: DataFrame, parameters: list, system: SystemConfiguration):
    param_dict = {param: 0 for param in parameters}
    for parameter in parameters:
      x1 = drop_unimportant_parameters(df1, [parameter], system)
      x2 = drop_unimportant_parameters(df2, [parameter], system)
      x1v = x1.groupby(parameter)["tps"].mean()
      x2v = x2.groupby(parameter)["tps"].mean()
      df = pd.merge(x1v, x2v, on=parameter, how="outer").fillna(0)
      x = df["tps_x"].values / df["tps_x"].values.sum()
      y = df["tps_y"].values / df["tps_y"].values.sum()
      distance = self.bhattacharyya_distance(x, y)
      param_dict[parameter] = distance
    return param_dict

  def bhattacharyya_distance(self, x, y):
    return -np.log(np.sum(np.sqrt(np.multiply(x, y))))
