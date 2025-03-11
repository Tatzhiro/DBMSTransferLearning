from abc import ABC, abstractmethod
from .utils import drop_unimportant_parameters
import pandas as pd
from pandas import DataFrame
import numpy as np
from .system_configuration import SystemConfiguration
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
  
  def compute_parameter_set_distance(self, df1: DataFrame, df2: DataFrame, parameters: list, system: SystemConfiguration):
    # TODO: drop_unimportant_parameters can raise an exception because df2 might not have enough samples
    x1 = drop_unimportant_parameters(df1, parameters, system)
    x2 = drop_unimportant_parameters(df2, parameters, system)
    x1v = x1.groupby(parameters)["tps"].mean()
    x2v = x2.groupby(parameters)["tps"].mean()
    df = pd.merge(x1v, x2v, on=parameters, how="inner")
    if len(df) <= 5:
      return np.inf
    x = df["tps_x"].values / df["tps_x"].values.sum()
    y = df["tps_y"].values / df["tps_y"].values.sum()
    distance = self.bhattacharyya_distance(x, y)
    
    # plt.figure()
    # plt.plot(df.index.map(str), x, alpha=0.5, label="x")
    # plt.plot(df.index.map(str), y, alpha=0.5, label="y")
    # plt.legend()
    # plt.show()
    # plt.close()
      
      
    return distance

  def bhattacharyya_distance(self, x, y):
    return -np.log(np.sum(np.sqrt(np.multiply(x, y))))


class SpearmanDistance(DistributionDistance):
  """
  Measure distance between two distributions using Spearman rank correlation.
  """
  def compute_distance(self, df1: DataFrame, df2: DataFrame, parameters: list, system: SystemConfiguration):
    param_dict = {param: 0 for param in parameters}
    for parameter in parameters:
      x1 = drop_unimportant_parameters(df1, [parameter], system)
      x2 = drop_unimportant_parameters(df2, [parameter], system)
      x1v = x1.groupby(parameter)["tps"].mean()
      x2v = x2.groupby(parameter)["tps"].mean()
      df = pd.merge(x1v, x2v, on=parameter, how="outer").fillna(0)
      x = df["tps_x"].values
      y = df["tps_y"].values
      distance = self.spearman_distance(x, y)
      param_dict[parameter] = distance
    return param_dict
  
  def compute_parameter_set_distance(self, df1: DataFrame, df2: DataFrame, parameters: list, system: SystemConfiguration):
    x1 = drop_unimportant_parameters(df1, parameters, system)
    x2 = drop_unimportant_parameters(df2, parameters, system)
    x1v = x1.groupby(parameters)["tps"].mean()
    x2v = x2.groupby(parameters)["tps"].mean()
    df = pd.merge(x1v, x2v, on=parameters, how="inner")
    if len(df) <= 5:
      return np.inf
    x = df["tps_x"].values
    y = df["tps_y"].values
    distance = self.spearman_distance(x, y)
    return distance
  
  def spearman_distance(self, x, y):
    """
    Range: [0, 2]
    0: perfect correlation
    1: no correlation
    2: perfect anti-correlation
    """
    return 1 - spearmanr(x, y).correlation