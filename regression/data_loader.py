from abc import ABC, abstractmethod
from typing import Union, Tuple
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from .graph import getNumCore, getMemSize
from .system_configuration import SystemConfiguration, LineairDBConfiguration
import random
from .jamshidi import L2SFeatureSelector, FeatureSelector
from .utils import set_unimportant_columns_to_one_value, drop_unimportant_parameters
from IPython import embed
from copy import deepcopy

class DataLoader(ABC):
  @abstractmethod
  def load_training_df(self, data_paths: list[str]) -> DataFrame:
    pass

  @abstractmethod
  def get_df_X(self, df):
    pass
  
  @abstractmethod
  def get_df_y(self, df):
    pass

  def load_training_data(self, data_paths: list[str], select_features: bool = False) -> (DataFrame, DataFrame):
    df = self.load_training_df(data_paths, select_features)
    return self.get_df_X(df), self.get_df_y(df)
  
  def preprocess_sampling_data(self, df):
    if self.important_features == None:
      self.important_features = self.feature_selector.select_important_features(df, self.system)
    df = drop_unimportant_parameters(df, self.important_features, self.system)
    return df

  def __init__(self, workload, system: SystemConfiguration = LineairDBConfiguration(), feature_selector: FeatureSelector = L2SFeatureSelector(), normalize: bool = False):
    self.workload: str = workload
    self.normalize: bool = normalize

    self.system: SystemConfiguration = system
    self.feature_selector: FeatureSelector = feature_selector

    self.important_features = None


class SimpleRegressionDataLoader(DataLoader):
  def get_df_X(self, df):
    return df.loc[:, [f"{self.system.get_perf_metric()}_x"]].values
  
  def get_df_y(self, df):
    return df[f"{self.system.get_perf_metric()}_y"]
  
  def load_training_df(self, data_paths: list[str], select_features: bool = False) -> DataFrame:
    df_list = [] 
    feature_is_selected = False
    for data_path in data_paths:
      df = self.drop_useless_columns(pd.read_csv(data_path))
      df = self.system.preprocess_param_values(df)
      if self.normalize:
        df = self.system.normalize_columns(df, data_path)

      if not feature_is_selected:
        important_features = self.system.get_param_names()
        if select_features == True:
          important_features = self.feature_selector.select_important_features(df, self.system)
          df = drop_unimportant_parameters(df, important_features, self.system)
          feature_is_selected = True

      important_features.sort()
      new_df = self.group_features(df, important_features)
      df_list.append(new_df)

    merge_df = pd.merge(df_list[0], df_list[1], on="config")
    merge_df["features"] = [important_features for _ in range(len(merge_df))] if feature_is_selected else "all"
    return merge_df
  
  def group_features(self, df, important_features):
    conf_df = deepcopy(df)
    conf_df["config"] = conf_df[important_features].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    series = conf_df.groupby("config")[self.system.get_perf_metric()].mean()
    return DataFrame({"config": series.keys(), self.system.get_perf_metric(): series.values})


  def drop_useless_columns(self, df: pd.DataFrame, columns_to_keep: list = []) -> pd.DataFrame:
    df = df[df["workload"] == self.workload]
    df = df[self.system.get_param_names() + columns_to_keep + [self.system.get_perf_metric()]]
    return df
  
  def __init__(self, workload, system: SystemConfiguration = LineairDBConfiguration(), feature_selector: FeatureSelector = L2SFeatureSelector(), normalize: bool = True):
    super().__init__(workload, system, feature_selector, normalize)
  

class MultivariateDataLoader(DataLoader):
  def get_df_X(self, df):
    features = self.system.get_features()
    if self.important_features != None:
      features = self.system.get_machine_specs() + self.important_features
    return df[features]
  
  def get_df_y(self, df):
    return df[self.system.get_perf_metric()]
  
  def load_training_df(self, data_paths: list[str], select_features: bool = False) -> DataFrame:
    """
    Loads dataframes with relevant columns and returns concat of them.
    No feature selection or preprocessing is done.
    """
    columns = self.system.get_features() + [self.system.get_perf_metric()]
    concat_df = pd.DataFrame(columns=columns)
    for filename in data_paths:
      df = self.make_data(filename)
      if len(concat_df) == 0: concat_df = df
      else: concat_df = pd.concat([concat_df, df])
    if select_features:
      concat_df = self.preprocess_sampling_data(concat_df)
    return concat_df
  
  def sample_target_data(self, data_path: str, sample_size: int = 0, select_features: bool = False) -> (DataFrame, DataFrame):
    df = self.sample_target_df(data_path, sample_size, select_features)
    return self.get_df_X(df), self.get_df_y(df)
  
  def sample_target_df(self, data_path: str, sample_size: int = 0, select_features: bool = False) -> (DataFrame, DataFrame):
    df = self.make_data(data_path)
    if select_features:
      df = self.preprocess_sampling_data(df)
    if sample_size == 0: return df
    seed = 0
    random.seed(seed)
    df = df.sample(n=sample_size, random_state=seed)
    return df
  
  def make_data(self, filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = self.system.preprocess_param_values(df)
    df = df[df["workload"] == self.workload][self.system.get_param_names() + [self.system.get_perf_metric()]]
    df = self.system.set_machine_specs(df, filename)
    if self.normalize:
      df = self.system.normalize_columns(df, filename)
    return df


