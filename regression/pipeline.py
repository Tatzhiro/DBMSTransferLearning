from regression.utils import read_data_csv
from regression.instance_similarity import InstanceSimilarity

class Pipeline:
    def __init__(self, context_retrieval, data_transfer=None, system=None, target_data_path=None, target_workload=None):
        self.context_retrieval = context_retrieval
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
        while(self.iter < run_iter):
            if not self.data_transfer.terminated:
                samples = self.data_transfer.sampled_data
                base_info = self.context_retrieval.retrieve_context(self.target_data_path, self.target_workload, samples)
                
                self.data_transfer.run_next_iteration(base_info)
            self.iter += 1
        self.data_transfer.fit()
            

    def predict(self):
        return self.data_transfer.predict(self.target_df)


class ContextRetrieval:
    def __init__(self, static_context_retrieval=None, dynamic_context_retrieval=None, min_samples=50):
        self.static_context_retrieval: InstanceSimilarity = static_context_retrieval
        self.static_similar_context_map = {}
        self.dynamic_context_retrieval = dynamic_context_retrieval
        self.min_samples = min_samples
        
    def retrieve_context(self, target_data_path, target_workload, samples=[]):
        if len(samples) < self.min_samples:
            key = (target_data_path, target_workload)
            if key in self.static_similar_context_map:
                return self.static_similar_context_map[key]
            else:
                self.static_similar_context_map[key] = \
                    self.static_context_retrieval.get_similar_datasets(target_data_path, target_workload, metadata=True)[0]
                return self.static_similar_context_map[key]
        else:
            return self.dynamic_context_retrieval.retrieve_context(samples)