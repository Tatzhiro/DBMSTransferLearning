from abc import ABC, abstractmethod
from pandas import DataFrame

from regression.utils import drop_unimportant_parameters
from regression.context_retrieval import ContextSimilarity


class DataTransfer(ABC):
    def __init__(self):
        self.seed: int = 0
        self.iter: int = 0
        self.terminated: bool = False
        
    @abstractmethod
    def initialize(self, base_df):
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
    
    @abstractmethod
    def receive_contexts(self, source_contexts: list[ContextSimilarity]) -> None:
        pass

    def simulate(self, run_iter=20):
        while(self.iter < run_iter):
            if not self.terminated:
                self.run_next_iteration()
            self.iter += 1
        self.fit()

    def calculate_mean_performance_groupedby_params(self, df: DataFrame, parameters: list[str]) -> DataFrame:
        df = df.groupby(self.system.get_param_names())[self.system.get_perf_metric()].mean().reset_index()
        df = drop_unimportant_parameters(df, parameters, self.system)
        return df
    