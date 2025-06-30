

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import LinearRegression

from regression.data_transfer import DataTransfer
from regression.context_retrieval import ContextSimilarity
from regression.utils import set_unimportant_columns_to_one_value


class ModelShift(DataTransfer):
    def __init__(self, system, model, feature_selector=None):
        self.feature_selector = feature_selector
        self.system = system
        self.model = model
        self.regression = None
        self.src_info_cache = {}
        self.df_to_model = {}
        super().__init__()
        

    class CacheEntry:
        def __init__(self, model, important_parameters, important_parameter_space):
            self.model = model
            self.important_parameters = important_parameters
            self.important_parameter_space = important_parameter_space
        

    def initialize(self, target_df):
        np.random.seed(self.seed)
        self.target_df = target_df
        self.target_data_population = target_df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
        self.sampled_data = pd.DataFrame()
        self.src_info_cache = {}
        self.important_parameters = None
        self.iter = 0
        self.terminated = False
        self.src_context = None
        
    
    def receive_contexts(self, source_contexts: list[ContextSimilarity]) -> None:
        source_contexts.sort(key=lambda x: x.similarity, reverse=True)
        self.src_context = source_contexts[0]
        

    def run_next_iteration(self) -> None:
        src_df = self.src_context.df
        src_wl = self.src_context.workload
        src_hw = self.src_context.hardware
        key = (src_wl, src_hw)
        if key in self.src_info_cache:
            cache_entry = self.src_info_cache[key]
            self.model = cache_entry.model
            self.important_parameters = cache_entry.important_parameters
            self.base_data = set_unimportant_columns_to_one_value(deepcopy(src_df), self.important_parameters, self.system)
            important_parameter_space = cache_entry.important_parameter_space
        else:
            important_parameters = self.select_important_parameters(src_df)
            self.important_parameters = important_parameters
            self.model.fit(src_df[important_parameters], src_df[self.system.get_perf_metric()])
            self.base_data = set_unimportant_columns_to_one_value(deepcopy(src_df), important_parameters, self.system)
            important_parameter_space = set_unimportant_columns_to_one_value(deepcopy(self.target_data_population), important_parameters, self.system)
            
            
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
        self.src_info_cache[key] = self.CacheEntry(self.model, self.important_parameters, important_parameter_space)
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
    