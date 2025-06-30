

import numpy as np
import pandas as pd
from copy import deepcopy

from regression.data_transfer import DataTransfer
from regression.context_retrieval import ContextSimilarity
from regression.jamshidi import L2SFeatureSelector
from regression.utils import epsilon_greedy, set_unimportant_columns_to_one_value


# class Proposed(TransferLearning):
#   def __init__(self, workload, system, target_data_path: str, base_data_path: str=None, ref_data_path: str=None, instance_similarity: InstanceSimilarity=None, regression_data_threshold=5, regression_rate=0.95) -> None:
#     self.distribution_distance = BhattacharyyaDistance()
#     self.distance_threshold = 0.001
#     self.feature_selector = ImportanceFeatureSelector()
    
#     if ref_data_path is None:
#       datasets: list[InstanceSimilarity.DatasetMetadata] = instance_similarity.get_similar_datasets(target_data_path, workload, n=2, metadata=True)
#       df = datasets[1].df
#       print(f"Using {datasets[1].workload_label}, {datasets[1].hardware_label} as reference data")
#       self.ref_df = df[system.get_param_names() + [system.get_perf_metric()]]
#       important_parameters = self.feature_selector.select_important_features(self.ref_df, system)
#       print(f"Important parameters: {important_parameters}")
#     else:
#       self.ref_data_path = ref_data_path
#       self.ref_df: DataFrame = read_data_csv(self.ref_data_path, self.system, self.workload)

#     self.regression_data_population: DataFrame
#     self.regression: LinearRegression
#     self.regression_data_samples: DataFrame
#     self.base_regression_data: DataFrame
#     self.regression_data_threshold = regression_data_threshold
#     self.regression_rate = regression_rate
    
#     self.dependent_param_data_population: DataFrame

#     self.machine_independent_parameters: list[str]

#     super().__init__(workload, system, instance_similarity, base_data_path, target_data_path)
#     self.parameters = list(set(self.select_important_parameters(self.ref_df)) | set(self.parameters))

  
#   def initialize(self):
#     np.random.seed(self.seed)
    
#     # regression rate: (rr*100)% chance to sample from the regression data population
#     self.rr = self.regression_rate
#     self.rthreshold = self.regression_data_threshold

#     # exploration-exploitation rate: (eer*100)% chance to explore
#     self.eer = 0.1

#     self.machine_independent_parameters = self.extract_machine_independent_parameters(self.parameters)

#     self.base_train_data = self.calculate_mean_performance_groupedby_params(self.base_df, self.parameters)
#     self.target_data_population = self.calculate_mean_performance_groupedby_params(self.target_df, self.system.get_param_names())

#     self.init_regression_data()

#     machine_dependent_params = [param for param in self.parameters if param not in self.machine_independent_parameters]
#     self.dependent_param_data_population = set_unimportant_columns_to_one_value(self.target_data_population, machine_dependent_params, self.system)
#     self.finetune_data = pd.DataFrame()

#     self.iter = 0
#     self.terminated = False
    
#     print(f"Machine independent parameters: {self.machine_independent_parameters}")
#     print(f"Machine dependent parameters: {machine_dependent_params}")
    
#   def calculate_mean_performance_groupedby_params(self, df: DataFrame, parameters: list[str]) -> DataFrame:
#     df = set_unimportant_columns_to_one_value(df, parameters, self.system)
#     df = df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
#     return df
    
#   def init_regression_data(self):
#     self.regression = LinearRegression()
#     self.regression_data_samples = pd.DataFrame()
    
  
#     if len(self.machine_independent_parameters) == 0: 
#       self.rr = 0
#     else:
#       self.regression_data_population = set_unimportant_columns_to_one_value(self.target_data_population, self.machine_independent_parameters, self.system)
#       self.base_regression_data = set_unimportant_columns_to_one_value(self.base_train_data, self.machine_independent_parameters, self.system)

#   def run_next_iteration(self) -> None:
#     sample_for_linear_coef = epsilon_greedy(self.rr)
#     if sample_for_linear_coef:
#       self.update_shift_coefficient()
#       if len(self.regression_data_samples) == self.rthreshold and self.rr > 0:
#         self.rr = (1 - self.rr)
#     else:
#       self.update_train_buffer()

#   def select_important_parameters(self, df=None) -> list[str]:
#     if df is None:
#       df = self.base_df
#     important_parameters = self.feature_selector.select_important_features(df, self.system)
#     return important_parameters
  
#   def extract_machine_independent_parameters(self, parameters) -> list[str]:
#     distances = self.distribution_distance.compute_distance(self.base_df, self.ref_df, parameters, self.system)
#     machine_independent_parameters = [param for param, distance in distances.items() if distance < self.distance_threshold]
#     return machine_independent_parameters
  
#   def fit(self) -> None:
#     base_train_data = deepcopy(self.base_train_data)
#     if len(self.regression_data_samples) > 2 and len(base_train_data) > 0:
      
#       # select parameters that should be linearly transformed
#       if len(self.machine_independent_parameters) > 0:
#         base_train_data = set_unimportant_columns_to_one_value(base_train_data, self.machine_independent_parameters, self.system)
        
#       base_train_data[self.system.get_perf_metric()] = self.regression.predict(base_train_data.loc[:, [self.system.get_perf_metric()]].values)
#       train_data = pd.concat([base_train_data, self.finetune_data])
#     else:
#       train_data = self.finetune_data
#     important_params = self.parameters
#     self.model.fit(train_data[important_params], train_data[self.system.get_perf_metric()])

#   def update_shift_coefficient(self) -> None:
#     merge_param = self.machine_independent_parameters 

#     while True:
#       if len(self.regression_data_population) == 0: 
#         print("No more linearly related data to sample")
#         self.rr = 0
#         return
      
#       sampled_row = self.regression_data_population.sample(n=1, random_state=self.seed)
#       self.regression_data_population = self.regression_data_population.drop(sampled_row.index)
      
#       if len(pd.merge(self.base_regression_data, sampled_row, on=merge_param)) == 0:
#         continue
      
#       # break immediately if regression data do not depend on base_train_data
#       if len(self.machine_independent_parameters) == 0:
#         break
      
#       # sampled_row may have been already sampled in update_train_buffer function
#       matching_row = self.base_train_data[self.base_train_data[self.parameters].eq(sampled_row[self.parameters].iloc[0]).all(axis=1)].index
#       if not matching_row.empty:
#         # drop the matching row from base_train_data and add it to finetune_data
#         self.base_train_data = self.base_train_data.drop(matching_row)
#         break

#     self.finetune_data = pd.concat([self.finetune_data, sampled_row])
      
#     self.regression_data_samples = pd.concat([self.regression_data_samples, sampled_row])

#     correlation_df = pd.merge(self.base_regression_data, self.regression_data_samples, on=merge_param)
#     if len(self.regression_data_samples) < len(correlation_df): raise UserWarning("Duplicate sampling")
#     self.regression.fit(correlation_df.loc[:, [f"{self.system.get_perf_metric()}_x"]].values, correlation_df[f"{self.system.get_perf_metric()}_y"])
#     # print(f"coef_: {self.regression.coef_}, intercept_: {self.regression.intercept_}")

#   def update_train_buffer(self) -> None:
#     while True:
#       if len(self.dependent_param_data_population) == 0 and len(self.target_data_population) == 0:
#         print("No more data to sample")
#         self.terminated = True
#         return
      
#       random_sampling = epsilon_greedy(self.eer)
#       if random_sampling or len(self.dependent_param_data_population) == 0:
#         sampled_row = self.target_data_population.sample(n=1, random_state=self.seed)
#         self.target_data_population = self.target_data_population.drop(sampled_row.index)
#       else:
#         sample_param = self.feature_selector.select_parameter_to_sample()
#         target_data_population = deepcopy(self.dependent_param_data_population)
#         param_population = set_unimportant_columns_to_one_value(target_data_population, [sample_param], self.system)
#         sampled_row = param_population.sample(n=1, random_state=self.seed)
#         self.dependent_param_data_population = self.dependent_param_data_population.drop(sampled_row.index)
        
      
#       duplicate = self.finetune_data[self.finetune_data.eq(sampled_row.iloc[0]).all(axis=1)]
#       if len(duplicate) == 0:
#         break

#     # matching_row contains the indices of the rows in base_train_data that have same values for all train_features as sampled_row
#     matching_row = self.base_train_data[self.base_train_data[self.parameters].eq(sampled_row[self.parameters].iloc[0]).all(axis=1)].index
#     if not matching_row.empty:
#       self.base_train_data = self.base_train_data.drop(matching_row)
#     self.finetune_data = pd.concat([self.finetune_data, sampled_row])

#   def predict(self, df: DataFrame) -> float:
#     important_params = self.parameters
#     df = self.system.preprocess_param_values(df)
#     return self.model.predict(df[important_params])