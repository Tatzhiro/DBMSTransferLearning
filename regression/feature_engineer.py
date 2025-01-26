from abc import abstractmethod, ABC
import numpy as np
from pandas import DataFrame
from IPython import embed
from scipy.special import expit
from copy import deepcopy
from .system_configuration import SystemConfiguration

class FeatureEngineer(ABC):
  @abstractmethod
  def modify_feature(self, df: DataFrame, system: SystemConfiguration) -> DataFrame:
    pass

class IdentityFeatureEngineer(FeatureEngineer):
  def modify_feature(self, df: DataFrame):
    return df
  
class LogCoreDistanceEngineer(FeatureEngineer):
  def modify_feature(self, df: DataFrame, system: SystemConfiguration):
    mod_df = deepcopy(df)
    mod_df.loc[:, "num_core"] = np.log(mod_df.loc[:, "num_core"].astype(float))
    return mod_df
  
class ThirdPolynomialEngineer(FeatureEngineer):
  def modify_feature(self, df: DataFrame, system: SystemConfiguration) -> DataFrame:
    mod_df = deepcopy(df)
    for param in system.get_param_names():
      mod_df[f"{param}^2"] = np.power(mod_df[f"{param}"], 2)
      mod_df[f"{param}^3"] = np.power(mod_df[f"{param}"], 3)
    return mod_df
    
class MachineIndependentThirdPolynomialEngineer(FeatureEngineer):
  def modify_feature(self, df: DataFrame, system: SystemConfiguration) -> DataFrame:
    mod_df = deepcopy(df)
    for param in system.get_machine_independent_parameters():
      mod_df[f"{param}^2"] = np.power(mod_df[f"{param}"], 2)
      mod_df[f"{param}^3"] = np.power(mod_df[f"{param}"], 3)
    return mod_df
  
class MachineDependentNormalizeEngineer(FeatureEngineer):
  def modify_feature(self, df: DataFrame, system: SystemConfiguration) -> DataFrame:
    mod_df = deepcopy(df)
    for param in system.get_machine_dependent_parameters():
      mod_df[f"normal_{param}"] = mod_df[param] / np.max(mod_df[param])
    return mod_df
  
class NoCoreEngineer(FeatureEngineer):
  def modify_feature(self, df: DataFrame, system: SystemConfiguration):
    mod_df = deepcopy(df)
    mod_df["num_core"] = 0
    return mod_df
  
class NoMachineSpecEngineer(FeatureEngineer):
  def modify_feature(self, df: DataFrame, system: SystemConfiguration):
    mod_df = deepcopy(df)
    for spec in system.get_machine_specs():
      mod_df[spec] = 0
    return mod_df
