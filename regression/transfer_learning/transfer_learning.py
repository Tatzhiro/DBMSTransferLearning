import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from regression.utils import drop_unimportant_parameters, read_data_csv
from regression.system_configuration import SystemConfiguration, LineairDBConfiguration
from regression.distribution_distance import BhattacharyyaDistance
from regression.jamshidi import LassoFeatureSelector, L2SFeatureSelector, FeatureSelector, ImportanceFeatureSelector
from regression.instance_similarity import ParameterImportanceSimilarity, InstanceSimilarity
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, Sum, ConstantKernel, Product
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression
from copy import deepcopy
from IPython import embed
from abc import ABC, abstractmethod

class TransferLearning(ABC):
    def __init__(self):
        self.seed: int = 0
        self.iter: int = 0
        self.terminated: bool = False
        
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def run_next_iteration(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, df: DataFrame) -> float:
        pass

    def simulate(self, run_iter=20):
        while(self.iter < run_iter):
            if not self.terminated:
                self.run_next_iteration()
            self.iter += 1
        self.fit()

    @abstractmethod
    def select_important_parameters(self) -> list[str]:
        pass
    
    def calculate_mean_performance_groupedby_params(self, df: DataFrame, parameters: list[str]) -> DataFrame:
        df = drop_unimportant_parameters(df, parameters, self.system)
        df = df.groupby(parameters)[self.system.get_perf_metric()].mean().reset_index()
        return df
    