

import numpy as np
import pandas as pd
from copy import deepcopy

from regression.data_transfer import DataTransfer
from regression.context_retrieval import ContextSimilarity
from regression.jamshidi import L2SFeatureSelector
from regression.utils import epsilon_greedy, set_unimportant_columns_to_one_value


class L2S(DataTransfer):
    def __init__(self, system, model, feature_selector=None):
        self.feature_selector = feature_selector if feature_selector is not None else L2SFeatureSelector(system)
        self.eer: float
        self.system = system
        self.model = model
        self.df_to_important_parameter_space = {}
        super().__init__()
        
    
    class CacheEntry:
        def __init__(self, important_parameter_space, important_parameters):
            self.important_parameter_space = important_parameter_space
            self.important_parameters = important_parameters
        

    def initialize(self, target_df):
        np.random.seed(self.seed)
        self.target_df = target_df
        self.target_data_population = target_df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
        self.sampled_data = pd.DataFrame()
        self.df_to_important_parameter_space = {}
        self.important_parameters = None
        self.eer = 0.1
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
        if key in self.df_to_important_parameter_space:
            entry = self.df_to_important_parameter_space[key]
            important_parameter_space = entry.important_parameter_space
            self.important_parameters = entry.important_parameters
        else:
            important_parameters = self.select_important_parameters(src_df)
            self.important_parameters = important_parameters
            important_parameter_space = set_unimportant_columns_to_one_value(deepcopy(self.target_data_population), important_parameters, self.system)
            print(f"Selected important parameters: {important_parameters}")
            
        while True:
            if len(important_parameter_space) == 0:
                print("No more data to sample")
                self.terminated = True
                return
            
            random_sampling = epsilon_greedy(self.eer)
            if random_sampling: 
                sampled_row = important_parameter_space.sample(n=1, random_state=self.seed)
            else:
                sample_param = self.feature_selector.select_parameter_to_sample()
                param_population = set_unimportant_columns_to_one_value(deepcopy(important_parameter_space), sample_param, self.system)
                sampled_row = param_population.sample(n=1, random_state=self.seed)
                
            self.target_data_population = self.target_data_population.drop(sampled_row.index)
            important_parameter_space = important_parameter_space.drop(sampled_row.index)
            duplicate = self.sampled_data[self.sampled_data.eq(sampled_row.iloc[0]).all(axis=1)]
            if len(duplicate) == 0:
                break
        self.df_to_important_parameter_space[key] = self.CacheEntry(important_parameter_space, self.important_parameters)
        self.sampled_data = pd.concat([self.sampled_data, sampled_row])

    def fit(self) -> None:
        important_params = self.important_parameters
        X = self.sampled_data[important_params]
        y = self.sampled_data[self.system.get_perf_metric()]
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> float:
        important_params = self.important_parameters
        df = self.system.preprocess_param_values(df)
        return self.model.predict(df[important_params])
  
    def select_important_parameters(self, df) -> list[str]:
        imporant_parameters = self.feature_selector.select_important_features(df, self.system)
        return imporant_parameters
    