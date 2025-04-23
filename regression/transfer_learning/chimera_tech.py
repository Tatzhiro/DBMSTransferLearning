from regression.proposed import TransferLearning
from regression.instance_similarity import InstanceSimilarity, ParameterImportanceSimilarity
from regression.jamshidi import ImportanceFeatureSelector, MultiImportanceFeatureSelector
from regression.distribution_distance import BhattacharyyaDistance, SpearmanDistance
from regression.utils import epsilon_greedy, set_unimportant_columns_to_one_value, read_data_csv

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor


class ChimeraTech(TransferLearning):
    def __init__(self, workload, system, target_data_path: str, instance_similarity: ParameterImportanceSimilarity):
        # self.distribution_distance = SpearmanDistance()
        # self.distance_threshold = 0.1
        
        self.feature_selector = None
        # self.parameter_importance = instance_similarity.get_parameter_importance(target_data_path, workload)
        super().__init__(workload, system, instance_similarity, None, target_data_path)
        
    def initialize(self):
        np.random.seed(self.seed)
        self.target_data_population = self.calculate_mean_performance_groupedby_params(self.target_df, self.parameters)
        self.finetune_data = pd.DataFrame()
        self.eer = 0.1
        self.iter = 0
        self.terminated = False
        
    
    def run_next_iteration(self):
        while True:
            if len(self.target_data_population) == 0:
                print("No more data to sample")
                self.terminated = True
                return
            
            random_sampling = epsilon_greedy(self.eer)
            if random_sampling: 
                sampled_row = self.target_data_population.sample(n=1, random_state=self.seed)
            else:
                sample_param = self.feature_selector.select_parameter_pair_to_sample()
                target_data_population = deepcopy(self.target_data_population)
                param_population = set_unimportant_columns_to_one_value(target_data_population, sample_param, self.system)
                sampled_row = param_population.sample(n=1, random_state=self.seed)
                
            self.target_data_population = self.target_data_population.drop(sampled_row.index)
            duplicate = self.finetune_data[self.finetune_data.eq(sampled_row.iloc[0]).all(axis=1)]
            if len(duplicate) == 0:
                break
        self.finetune_data = pd.concat([self.finetune_data, sampled_row])
        
    
    # def reuse_data(self):
    #     # (df, workload, hardware, kl divergence, parameter importance)
    #     similar_datasets = self.similar_datasets
    #     transformed_data = pd.DataFrame()
        
    #     for i, p in enumerate(self.system.get_param_names()):
    #         for j, q in enumerate(self.system.get_param_names()):
    #             if i > j:
    #                 continue
    #             parameter_set = list(set([p, q]))
    #             min_distance = self.distance_threshold
    #             min_src = None
    #             for dataset in similar_datasets:
    #                 workload = dataset.workload_label
    #                 hardware = dataset.hardware_label
    #                 df = read_data_csv(f"dataset/transfer_learning/mysql/chimera_tech/{hardware}-result.csv", self.system, workload)
    #                 source_df = self.system.preprocess_param_values(df)
    #                 try:
    #                     distance = self.distribution_distance.compute_parameter_set_distance(source_df, self.finetune_data, parameter_set, self.system)
    #                 except UserWarning:
    #                     distance = np.inf
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     min_src = source_df
    #                     min_dataset = dataset

    #             if min_src is None:
    #                 continue
                
    #             data = self.linear_transform(min_src, self.finetune_data, parameter_set)
    #             print(f"Transforming data using {parameter_set} from {min_dataset.workload_label}, {min_dataset.hardware_label}")
    #             # drop any rows that have same parameter values as any row in the finetune data
    #             mask = data[self.system.get_param_names()].apply(tuple, axis=1).isin(self.finetune_data[self.system.get_param_names()].apply(tuple, axis=1))
    #             data = data[~mask]
    #             transformed_data = pd.concat([transformed_data, data])

    #     return transformed_data
                
                
    # def linear_transform(self, source_df, target_df, parameter_set):
    #     metric = self.system.get_perf_metric()
        
    #     source_df = self.system.preprocess_param_values(source_df)
    #     source_df = source_df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
    #     target_df = self.system.preprocess_param_values(target_df)
        
    #     source_series = source_df.groupby(parameter_set)[metric].mean()
    #     target_series = target_df.groupby(parameter_set)[metric].mean()
        
    #     df = pd.merge(source_series, target_series, on=parameter_set)
    #     reg = RANSACRegressor().fit(df[[f"{metric}_x"]], df[f"{metric}_y"])
        
    #     source_df = set_unimportant_columns_to_one_value(source_df, parameter_set, self.system)
    #     source_df[metric] = reg.predict(source_df[metric].values.reshape(-1, 1))
        
    #     # plt.subplot(2, 1, 1)
    #     # plt.plot(df.index.map(str), df[f"{metric}_x"], alpha=0.5, label="x")
    #     # plt.plot(df.index.map(str), df[f"{metric}_y"], alpha=0.5, label="y")
    #     # plt.legend()
    #     # plt.subplot(2, 1, 2)
        
    #     # plt.plot(source_df.index.map(str), source_df[metric], alpha=0.5, label="transformed")
    #     # plt.xticks(rotation=45)
    #     # plt.show()
    #     # plt.close()
        
    #     return source_df
    
            
    def fit(self) -> None:
        params = self.parameters
        target = self.system.get_perf_metric()
        
        X = self.finetune_data[params]
        y = self.finetune_data[target]
        
        # past_data = self.reuse_data()
        
        # if len(past_data) > 0:
        #     X = pd.concat([X, past_data[params]])
        #     y = pd.concat([y, past_data[target]])
        
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> float:
        important_params = self.parameters
        df = self.system.preprocess_param_values(df)
        return self.model.predict(df[important_params])

        
    def select_important_parameters(self):
        if self.feature_selector is None:
            base_df = self.calculate_mean_performance_groupedby_params(self.base_df, self.system.get_param_names())
            self.feature_selector = MultiImportanceFeatureSelector(base_df, self.system)
        return self.feature_selector.select_important_features()
