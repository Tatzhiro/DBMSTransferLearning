from .data_loader import MultivariateDataLoader, SimpleRegressionDataLoader, DataLoader
from .feature_engineer import LogCoreDistanceEngineer, FeatureEngineer, IdentityFeatureEngineer, NoCoreEngineer, NoMachineSpecEngineer
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import random
from .jamshidi import LassoFeatureSelector
from .utils import set_unimportant_columns_to_one_value, drop_unimportant_parameters
from IPython import embed
from copy import deepcopy

class Pipeline(ABC):
  @abstractmethod
  def fit(self, data_paths: list[str]) -> None:
    pass

  @abstractmethod
  def fit_df_internal(self, df: DataFrame) -> None:
    pass

  @abstractmethod
  def predict(self, df: DataFrame) -> ndarray:
    pass

  @abstractmethod
  def cross_validation(self, train_data_path: str, target_data_path: str, n_splits: int = 5, train_size=20):
    pass

  def calculate_validation_error(self, test_df: DataFrame) -> float:
    X = self.data_loader.get_df_X(test_df)
    y_pred = self.predict(X)
    y_true = self.data_loader.get_df_y(test_df)
    return mean_absolute_percentage_error(y_true, y_pred)
  
  def set_model(self, model):
    self.model = make_pipeline(StandardScaler(), model)

  def __init__(self, data_loader: DataLoader, model, select_features = False) -> None:
    self.data_loader: DataLoader = data_loader
    self.model = make_pipeline(StandardScaler(), model)
    self.select_features: bool = select_features


class SimpleRegressionPipeline(Pipeline):
  def fit(self, data_paths: list[str], sample_size: int = 10) -> None:
    if len(data_paths) > 2:
      data_paths = data_paths[-2:]

    df = self.data_loader.load_training_df(data_paths, self.select_features)
    seed = 0
    random.seed(seed)
    df = df.sample(n=sample_size, random_state=seed)

    self.fit_df_internal(df)

  def fit_df_internal(self, df: DataFrame) -> None:
    X = self.data_loader.get_df_X(df)
    y = self.data_loader.get_df_y(df)
    self.model.fit(X, y)

  def predict(self, df: DataFrame) -> ndarray:
    return self.model.predict(df)

  def cross_validation(self, train_data_path: str, target_data_path: str, n_splits: int = 5, train_size=20):
    if type(train_data_path) != str:
      train_data_path = train_data_path[-1]
    df = self.data_loader.load_training_df([train_data_path, target_data_path], self.select_features)
    seed = 0
    ss = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)
    score = []
    for (train_index, test_index) in ss.split(df):
      train_df = df.iloc[train_index]
      test_df = df.iloc[test_index]
      self.fit_df_internal(train_df)
      score = np.append(score, self.calculate_validation_error(test_df))
    return score.mean()
  
  def __init__(self, data_loader: SimpleRegressionDataLoader, model = LinearRegression(), select_features: bool = False) -> None:
    super().__init__(data_loader, model, select_features)



class MultivariateRegressionPipeline(Pipeline):
  def fit(self, data_paths: list[str]) -> None:
    df = self.data_loader.load_training_df(data_paths, self.select_features)
    self.fit_df_internal(df)

  def fit_df_internal(self, df: DataFrame) -> None:
    X = self.data_loader.get_df_X(df)
    y = self.data_loader.get_df_y(df)
    X = self.modify_features(X)
    self.model.fit(X, y)

  def fit_with_target(self, source_envs: list[str], target_env: str, sample_size: int) -> None:
    src_X, src_y = self.data_loader.load_training_data(source_envs, self.select_features)
    src_X = self.modify_features(src_X)

    tgt_X, tgt_y = self.data_loader.sample_target_data(target_env, sample_size, self.select_features)
    tgt_X = self.modify_features(tgt_X)

    X = pd.concat([src_X, tgt_X])
    y = pd.concat([src_y, tgt_y])
    self.model.fit(X, y)

  def predict(self, df: DataFrame) -> ndarray:
    X = self.data_loader.get_df_X(df)
    X = self.modify_features(X)
    return self.model.predict(X)
  
  def modify_features(self, X: DataFrame):
    modified_X = deepcopy(X)
    for fe in self.feature_engineers:
      modified_X = fe.modify_feature(modified_X, self.data_loader.system)
    return modified_X
  
  def cross_validation(self, train_data_paths: list, target_data_path: str, n_splits: int = 5, train_size=20):
    if train_size == 0:
      domain_df = self.data_loader.load_training_df(train_data_paths, self.select_features)
      target_df = self.data_loader.load_training_df([target_data_path], self.select_features)
      self.fit_df_internal(domain_df)
      score = self.calculate_validation_error(target_df)
      return score
    
    domain_df = self.data_loader.load_training_df(train_data_paths, self.select_features)
    target_df = self.data_loader.sample_target_df(target_data_path, sample_size=0, select_features=self.select_features)
    seed = 0
    ss = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)
    score = []
    for (train_index, test_index) in ss.split(target_df):
      train_df, test_df = self.__split_df(domain_df, target_df, train_index, test_index)
      self.fit_df_internal(train_df)
      score = np.append(score, self.calculate_validation_error(test_df))
    return score.mean()
  
  def test_model(self, train_data_paths: list, target_data_path: str, n_splits: int = 5, train_size=20):
    raise Exception("Where is this even called from?")
    if train_size == 0:
      domain_df = self.data_loader.load_training_df(train_data_paths)
      target_df = self.data_loader.load_training_df([target_data_path])
      self.fit_df_internal(domain_df)
      score = self.calculate_validation_error(target_df)
      return score
    
    domain_df = self.data_loader.load_training_df(train_data_paths)
    val_df = self.data_loader.load_training_df([target_data_path])
    
    sample_df = deepcopy(val_df)
    sample_df = self.data_loader.preprocess_sampling_data(sample_df)

    seed = 0
    ss = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)
    score = []
    for (train_index, _) in ss.split(sample_df):
      train_df = pd.concat([domain_df, sample_df.iloc[train_index]])
      test_df = pd.concat([sample_df, val_df]).drop_duplicates(keep=False)
      self.fit_df_internal(train_df)
      score = np.append(score, self.calculate_validation_error(test_df))
    return score.mean()
  
  def __split_df(self, domain_df: DataFrame, target_df: DataFrame, train_index: ndarray, test_index: ndarray) -> (DataFrame, DataFrame):
    train_df = pd.concat([domain_df, target_df.iloc[train_index]])
    for clm in train_df.columns: train_df[clm] = pd.to_numeric(train_df[clm])
    test_df = target_df.iloc[test_index]
    return train_df, test_df

  def __init__(self, data_loader: MultivariateDataLoader, feature_engineers: list[FeatureEngineer] = [IdentityFeatureEngineer()], model = LinearRegression(), select_features: bool = False) -> None:
    self.feature_engineers: list[FeatureEngineer] = feature_engineers
    super().__init__(data_loader, model, select_features)

class LinearShiftMultiRegressionPipeline(MultivariateRegressionPipeline):
  def fit(self, data_paths: list[str]) -> None:
    df = self.data_loader.load_training_df(data_paths)
    self.a = self.get_factor(df)
    self.fit_df_internal(df)

  def fit_df_internal(self, df: DataFrame) -> None:
    X = self.data_loader.get_df_X(df)
    y = self.data_loader.get_df_y(df) / self.scale_factor(df["num_core"])
    X = self.modify_features(X)
    self.model.fit(X, y)

  def get_factor(self, df: DataFrame):
    machine_power = df["num_core"].drop_duplicates().sort_values().to_list()
    self.N = machine_power[1] / machine_power[0]

    features = self.fs.select_important_features(df, self.system)
    df = set_unimportant_columns_to_one_value(df, features, self.system)
    df_x = df[df["num_core"] == machine_power[0]]
    df_y = df[df["num_core"] == machine_power[1]]
    regression_df = pd.merge(df_x, df_y, on=self.system.get_param_names())
    for machine_param in self.system.get_machine_dependent_parameters():
      regression_df = regression_df[regression_df[machine_param] == np.min(regression_df[machine_param])]
    X = regression_df[f"{self.system.get_perf_metric()}_x"]
    y = regression_df[f"{self.system.get_perf_metric()}_y"]
    factor = (y / X).median()
    return factor
  
  def scale_factor(self, core):
    return np.power(self.a, np.emath.logn(self.N, core/self.N) - 1)

  def fit_with_target(self, source_envs: list[str], target_env: str, sample_size: int) -> None:
    pass

  def predict(self, df: DataFrame) -> ndarray:
    params = self.modify_features(df)
    y = self.model.predict(params) * self.scale_factor(df["num_core"])
    return y
  
  def modify_features(self, X: DataFrame):
    modified_X = deepcopy(X)
    for fe in self.feature_engineers:
      modified_X = fe.modify_feature(modified_X, self.data_loader.system)
    return modified_X
  
  def cross_validation(self, train_data_paths: list, target_data_path: str, n_splits: int = 5, train_size=20):
    pass

  def test_model(self, train_data_paths: list, target_data_path: str, n_splits: int = 5, train_size=20):
    pass

  def __init__(self, data_loader: MultivariateDataLoader, feature_engineers: list[FeatureEngineer] = [NoMachineSpecEngineer()], model = LinearRegression(), select_features: bool = False) -> None:
    self.a = 1
    self.N = 0
    self.fs = data_loader.feature_selector
    self.system = data_loader.system
    self.simple_regression_data_loader = SimpleRegressionDataLoader(data_loader.workload, data_loader.system, data_loader.feature_selector, True)
    super().__init__(data_loader, feature_engineers, model, select_features)
