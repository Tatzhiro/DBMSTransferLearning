from regression.utils import read_data_csv
from regression.context_retrieval import ContextRetrieval


class Pipeline:
    """
    A standard transfer learning pipeline.
    """
    def __init__(self, context_retrieval, data_transfer=None, system=None, target_data_path=None, target_workload=None):
        self.context_retrieval: ContextRetrieval = context_retrieval
        self.data_transfer = data_transfer
        self.system = system
        self.target_data_path = target_data_path
        self.target_workload = target_workload
        self.target_df = read_data_csv(target_data_path, system, target_workload)
        
        self.seed = 0
        self.iter = 0
        
                       
    def initialize(self):
        self.data_transfer.seed = self.seed
        self.data_transfer.initialize(self.target_df)
        self.iter = 0
            
    
    def simulate(self, run_iter=20):
        if self.data_transfer.terminated:
            self.iter += run_iter
            return
        
        while(self.iter < run_iter):
            if not self.data_transfer.terminated:
                self.load_source_contexts()
                self.data_transfer.run_next_iteration()
            self.iter += 1
        
        self.data_transfer.fit()
        
        
    def load_source_contexts(self):
        samples = self.data_transfer.sampled_data
        base_info = self.context_retrieval.retrieve_contexts(self.target_data_path, self.target_workload, samples)
        self.data_transfer.receive_contexts(base_info)
            

    def predict(self):
        return self.data_transfer.predict(self.target_df)


class FastPipeline(Pipeline):
    """
    A fast version of the Pipeline that avoids loading contexts at each iteration.
    Can only be used with data transfer methods that do not require context loading at each iteration.
    It loads contexts once at the END of `simulate` method.
    """
    def simulate(self, run_iter=20):
        if self.data_transfer.terminated:
            self.iter += run_iter
            return
        
        while(self.iter < run_iter):
            if not self.data_transfer.terminated:
                self.data_transfer.run_next_iteration()
            self.iter += 1
        
        self.load_source_contexts()
        self.data_transfer.fit()
