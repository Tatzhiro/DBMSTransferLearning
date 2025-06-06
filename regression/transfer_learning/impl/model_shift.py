

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from regression.transfer_learning.transfer_learning import TransferLearning
from regression.instance_similarity import InstanceSimilarity
from regression.jamshidi import L2SFeatureSelector
from regression.utils import epsilon_greedy, set_unimportant_columns_to_one_value, drop_unimportant_parameters


class ModelShift(TransferLearning):
    def __init__(self, system, model, feature_selector=None):
        self.feature_selector = feature_selector
        self.system = system
        self.model = model
        self.regression = None
        self.src_info_cache = {}
        self.df_to_model = {}
        super().__init__()
        
    class CacheEntry:
        def __init__(self, model, important_parameters):
            self.model = model
            self.important_parameters = important_parameters
        

    def initialize(self, target_df):
        np.random.seed(self.seed)
        self.target_df = target_df
        self.target_data_population = target_df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
        self.sampled_data = pd.DataFrame()
        self.src_info_cache = {}
        self.important_parameters = None
        self.eer = 0.1
        self.iter = 0
        self.terminated = False


    def run_next_iteration(self, base_metadata: InstanceSimilarity.DatasetMetadata) -> None:
        base_df = base_metadata.df
        base_wl = base_metadata.workload_label
        base_hw = base_metadata.hardware_label
        key = (base_wl, base_hw)
        if key in self.src_info_cache:
            cache_entry = self.src_info_cache[key]
            self.model = cache_entry.model
            self.important_parameters = cache_entry.important_parameters
        else:
            important_parameters = self.select_important_parameters(base_df)
            self.model.fit(base_df[important_parameters], base_df[self.system.get_perf_metric()])
            self.important_parameters = important_parameters
            self.src_info_cache[key] = self.CacheEntry(self.model, important_parameters)
            self.base_data = set_unimportant_columns_to_one_value(deepcopy(base_df), important_parameters, self.system)
            self.target_data_population = set_unimportant_columns_to_one_value(deepcopy(self.target_data_population), important_parameters, self.system)
            
            
            
        important_parameter_space = self.target_data_population
        while True:
            if len(important_parameter_space) == 0:
                print("No more data to sample")
                self.terminated = True
                return
            
            sampled_row = important_parameter_space.sample(n=1, random_state=self.seed)
                
            self.target_data_population = self.target_data_population.drop(sampled_row.index)
            important_parameter_space = important_parameter_space.drop(sampled_row.index)
            duplicate = self.sampled_data[self.sampled_data.eq(sampled_row.iloc[0]).all(axis=1)]
            if len(duplicate) == 0:
                break
        self.sampled_data = pd.concat([self.sampled_data, sampled_row])

    def fit(self) -> None:
        important_params = self.important_parameters
        
        regression_data = pd.merge(self.base_data, self.sampled_data, on=important_params, how='inner')
        if len(regression_data) > 2:
            X = regression_data.loc[:, [f"{self.system.get_perf_metric()}_x"]].values
            y = regression_data[f"{self.system.get_perf_metric()}_y"].values
            self.regression = LinearRegression().fit(X, y)
        else:
            self.regression = None


    def predict(self, df: pd.DataFrame) -> float:
        important_params = self.important_parameters
        df = self.system.preprocess_param_values(df)
        prediction = self.model.predict(df[important_params]).reshape(-1, 1)
        if self.regression is not None:
            prediction = self.regression.predict(prediction)
        return prediction

  
    def select_important_parameters(self, df) -> list[str]:
        if self.feature_selector is None:
            return self.system.get_param_names()
        
        imporant_parameters = self.feature_selector.select_important_features(df, self.system)
        return imporant_parameters
    