import pandas as pd
import statsmodels.api as smf
from statsmodels.regression.linear_model import PredictionResults, RegressionResults
import random
from abc import ABC, abstractmethod
from typing import Union, Tuple

from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import numpy as np
from statsmodels.regression.linear_model import RegressionResults
from .graph import getNumCore, getMemSize
from .system_configuration import SystemConfiguration, LineairDBConfiguration
from .feature_engineer import FeatureEngineer, LogCoreDistanceEngineer
from .jamshidi import L2SFeatureSelector
from .utils import set_unimportant_columns_to_one_value
from IPython import embed

from numpy import ndarray
from pandas import DataFrame

class Model(ABC):
  def validation(self, train_data_paths: Union[list, str], target_data_path: str):
    domain_df, target_df = self.make_domain_target_data(train_data_paths, target_data_path)
    self.fit_model(domain_df)
    score = self.calculate_validation_error(target_df)
    return score

  def cross_validation(self, train_data_paths: Union[list, str], target_data_path: str, n_splits: int = 5, train_size=20):
    domain_df, target_df = self.make_domain_target_data(train_data_paths, target_data_path)
    seed = 0
    ss = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)
    score = []
    for (train_index, test_index) in ss.split(target_df):
      train_df, test_df = self.split_df(domain_df, target_df, train_index, test_index)
      self.fit_model(train_df)
      score = np.append(score, self.calculate_validation_error(test_df))
    return score.mean()

  @abstractmethod
  def make_domain_target_data(self, train_data_paths: Union[list, str], target_data_path: str) -> Tuple[DataFrame, DataFrame]:
    pass

  @abstractmethod
  def split_df(self, domain_df: DataFrame, target_df: DataFrame, train_index: ndarray, test_index: ndarray) -> (DataFrame, DataFrame):
    pass

  @abstractmethod
  def calculate_validation_error(self, test_df: DataFrame) -> float:
    pass

  def make_model(self, train_data_paths: list, target_data_path: str, sample_size: int = 10):
    domain_df, target_df = self.make_domain_target_data(train_data_paths, target_data_path)
    if sample_size == 0: return self.fit_model(domain_df)

    seed = 0
    random.seed(seed)
    target_df = target_df.sample(n=sample_size, random_state=seed)
    concat_df: DataFrame = pd.concat([domain_df, target_df])
    self.fit_model(concat_df)

  @abstractmethod
  def fit_model(self, df: pd.DataFrame):
    pass

  @abstractmethod
  def predict(self, df: pd.DataFrame) -> ndarray:
    pass

class MLModel(Model):
  def make_domain_target_data(self, train_data_paths: list, target_data_path: str = None) -> pd.DataFrame:
    columns = self.system.get_features() + [self.system.get_perf_metric()]
    merge_df = pd.DataFrame(columns=columns)
    for filename in train_data_paths:
      df = self.make_data(filename)
      merge_df = pd.concat([merge_df, df])
    if target_data_path == None: return merge_df

    target_df = self.make_data(target_data_path)

    if self.l2s != None: 
      # WANTFIX: its better if we can remove set_non_normalized_default_param_value from here
      important_params = self.l2s.select_important_features(df)
      target_df = set_unimportant_columns_to_one_value(df, important_params, self.system.get_param_names())
    return (merge_df, target_df)
  
  def split_df(self, domain_df: DataFrame, target_df: DataFrame, train_index: ndarray, test_index: ndarray) -> (DataFrame, DataFrame):
    train_df = pd.concat([domain_df, target_df.iloc[train_index]])
    for clm in train_df.columns: train_df[clm] = pd.to_numeric(train_df[clm])
    test_df = target_df.iloc[test_index]
    return train_df, test_df
  
  def calculate_validation_error(self, test_df: DataFrame) -> float:
    y_pred = self.model.predict(test_df[self.system.get_features()])
    y_true = test_df[self.system.get_perf_metric()]
    return mean_absolute_percentage_error(y_true, y_pred)

  def make_data(self, filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df[df["workload"] == self.workload][self.system.get_param_names() + [self.system.get_perf_metric()]]
    # TODO: add machine spec info in SystemConfiguration class
    num_core = getNumCore(filename)
    df["num_core"] = num_core
    # df = df[df["clients"] < num_core]
    if self.normalize:
      df = self.system.normalize_columns(df, filename)

    if self.feature_engineer != None:
      df = self.feature_engineer.modify_feature(df)
    return df
  
  def fit_model(self, df: DataFrame) -> LinearModel:
    X = df[self.system.get_features()]
    y = df[self.system.get_perf_metric()]
    self.model.fit(X, y)
    # print(f"coefficient: {dict(zip(X, self.model.get_params()['linearregression'].coef_))}")

  def predict(self, df: DataFrame) -> ndarray:
    if self.feature_engineer != None:
      df = self.feature_engineer.modify_feature(df)
    return self.model.predict(df)

  def __init__(self, workload, system, model = LassoCV(), l2s: bool = False, feature_engineer = None, normalize: bool = False):
    self.workload: str = workload
    self.system: SystemConfiguration = system
    self.model = make_pipeline(StandardScaler(), model)
    self.l2s: L2SFeatureSelector = L2SFeatureSelector(system) if l2s else None
    self.feature_engineer: FeatureEngineer = feature_engineer
    self.normalize: bool = normalize

class SMLinearModel(MLModel):
  def fit_model(self, df: DataFrame):
    X = df[self.system.get_features()]
    y = df[self.system.get_perf_metric()]
    self.model = smf.OLS(y, X).fit()

  def predict(self, X, return_std=True) -> (np.ndarray, np.ndarray, np.ndarray):
    pred_results: PredictionResults = self.model.get_prediction(X)
    mean = pred_results.predicted_mean
    if return_std:
      conf_interval = pred_results.conf_int()
      return mean, conf_interval[:, 0], conf_interval[:, 1]
    return mean

  def _confidence_interval(self, X):
    low = np.matmul(X, (self.model.coef_ - 1.96 * self.variance.diagonal())) + self.model.intercept_
    high = np.matmul(X, (self.model.coef_ + 1.96 * self.variance.diagonal())) + self.model.intercept_
    return high, low
  
  def _calculate_variance(self, X, y):
    predictor = self.model.predict(X)
    residual = y - predictor
    std_est = (1 / (X.shape[0] - X.shape[1] - 1)) * np.matmul(residual.T, residual)
    coef = np.linalg.inv(np.matmul(X.T, y))
    variance = coef * std_est
    return variance

  def __init__(self, workload, system: SystemConfiguration):
    self.model: RegressionResults
    super().__init__(workload, system, model=None)

class ValovModel(Model):
  def split_df(self, source_df: DataFrame, target_df: DataFrame, train_index: ndarray, test_index: ndarray) -> (DataFrame, DataFrame):
    train_df = pd.merge(source_df, target_df.iloc[train_index], on='config', how='inner')
    test_df = pd.merge(source_df, target_df.iloc[test_index], on='config', how='inner')
    return train_df, test_df

  def make_domain_target_data(self, train_data_path: str, target_data_path: str, get_important_params: bool = False) -> (DataFrame, DataFrame):
    source_df = self.preprocess(train_data_path)
    target_df = self.preprocess(target_data_path)

    important_params = self.l2s.select_important_features(source_df) if self.reduce_features else self.system.get_param_names()

    source_df = self.extract_columns(source_df, important_params)
    target_df = self.extract_columns(target_df, important_params)
    if get_important_params: return source_df, target_df, important_params
    return source_df, target_df
  
  def make_data(self, data_path, parameters):
    df = self.preprocess(data_path)
    df = self.extract_columns(df, parameters)
    return df
    
  def make_model(self, train_data_path: str, target_data_path: str, sample_size: int = 10):
    source_df, target_df = self.make_domain_target_data(train_data_path, target_data_path)
    if sample_size == 0: return self.fit_model(source_df)

    seed = 0
    random.seed(seed)
    target_df = target_df.sample(n=sample_size, random_state=seed)
    merge_df = pd.merge(source_df, target_df, on='config', how='inner')
    self.fit_model(merge_df)
  
  def fit_model(self, df: DataFrame):
    x_perf = df.loc[:, [f"{self.system.get_perf_metric()}_x"]].values
    y_perf = df.loc[:, f"{self.system.get_perf_metric()}_y"].values
    self.model.fit(x_perf, y_perf)

  def predict(self, df: DataFrame) -> ndarray:
    return self.model.predict(df)
  
  def calculate_validation_error(self, test_df: DataFrame) -> float:
    x_label = f"{self.system.get_perf_metric()}_x"
    y_label = f"{self.system.get_perf_metric()}_y"
    y_pred = self.predict(test_df.loc[:, [x_label]].values)
    y_true = test_df[y_label]
    return mean_absolute_percentage_error(y_true, y_pred)
  
  def preprocess(self, filename):
    df = self.drop_useless_columns(pd.read_csv(filename))
    df = self.system.normalize_columns(df, filename)
    return df
  
  def drop_useless_columns(self, df: pd.DataFrame, columns_to_keep: list = []) -> pd.DataFrame:
    df = df[df["workload"] == self.workload]
    df = df[self.system.get_param_names() + columns_to_keep + [self.system.get_perf_metric()]]
    return df

  def extract_columns(self, df: pd.DataFrame, important_params: list) -> pd.DataFrame:
    df = set_unimportant_columns_to_one_value(df, important_params, self.system.get_param_names())
    df["config"] = df[important_params].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    df = df.groupby("config")[self.system.get_perf_metric()].mean()
    return df

  def __init__(self, workload: str, system: SystemConfiguration, reduce_features: bool = False, model = LinearRegression()) -> None:
    self.model: LinearRegression = model
    self.workload: str = workload
    self.system: SystemConfiguration = system
    self.reduce_features: bool = reduce_features
    self.l2s: L2SFeatureSelector = L2SFeatureSelector(system)