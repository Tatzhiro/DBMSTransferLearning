

import numpy as np
import pandas as pd
from copy import deepcopy

from regression.proposed import TransferLearning
from regression.jamshidi import L2SFeatureSelector
from regression.utils import epsilon_greedy, set_unimportant_columns_to_one_value


class L2S(TransferLearning):
    def __init__(self, system, target_df, base_df, model, feature_selector=None):
        self.feature_selector = feature_selector if feature_selector is not None else L2SFeatureSelector(system)
        self.eer: float
        self.model = model
        self.target_df = target_df
        self.base_df = base_df
        
        # reduce target sample space to the important parameter space of nearest context
        self.important_parameters = self.feature_selector.select_important_features(base_df)
        self.initialize()
        super().__init__()
        

    def initialize(self):
        np.random.seed(self.seed)
        self.target_data_population = self.calculate_mean_performance_groupedby_params(self.target_df, self.parameters)
        self.finetune_data = pd.DataFrame()
        self.eer = 0.1
        self.iter = 0
        self.terminated = False

    def run_next_iteration(self) -> None:
        while True:
            if len(self.target_data_population) == 0:
                print("No more data to sample")
                self.terminated = True
                return
            
            random_sampling = epsilon_greedy(self.eer)
            if random_sampling: 
                sampled_row = self.target_data_population.sample(n=1, random_state=self.seed)
            else:
                sample_param = self.feature_selector.select_parameter_to_sample()
                target_data_population = deepcopy(self.target_data_population)
                param_population = set_unimportant_columns_to_one_value(target_data_population, [sample_param], self.system)
                sampled_row = param_population.sample(n=1, random_state=self.seed)
                
            self.target_data_population = self.target_data_population.drop(sampled_row.index)
            duplicate = self.finetune_data[self.finetune_data.eq(sampled_row.iloc[0]).all(axis=1)]
            if len(duplicate) == 0:
                break
            self.finetune_data = pd.concat([self.finetune_data, sampled_row])

    def fit(self) -> None:
        important_params = self.parameters
        X = self.finetune_data[important_params]
        y = self.finetune_data[self.system.get_perf_metric()]
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> float:
        important_params = self.parameters
        df = self.system.preprocess_param_values(df)
        return self.model.predict(df[important_params])
  