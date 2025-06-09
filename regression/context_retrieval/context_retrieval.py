import os


class Context:
    def __init__(self, df, workload, hardware):
        """
        Represents a context containing a DataFrame, workload, and optional hardware information.
        :param df: DataFrame containing the dataset.
        :param workload: Workload label for the dataset.
        :param hardware: Hardware label for the dataset, in form of f"{num_core}c{mem_size}g-result.csv".
        """
        self.df = df
        self.workload = workload
        self.hardware = hardware

    def __hash__(self):
        return hash((self.data_path, self.workload, self.hardware))

    def __eq__(self, other):
        return (self.data_path, self.workload, self.hardware) == (other.data_path, other.workload, other.hardware)
    
    
class ContextSimilarity(Context):
    def __init__(self, df, workload, hardware, distance, similarity):
        super().__init__(df, workload, hardware)
        self.distance = distance
        self.similarity = similarity


class ContextRetrieval:
    def __init__(self, static_context_retrieval=None, dynamic_context_retrieval=None, min_samples=50):
        self.static_context_retrieval = static_context_retrieval
        self.static_similar_context_map = {}
        self.dynamic_context_retrieval = dynamic_context_retrieval
        self.min_samples = min_samples


    def retrieve_context(self, target_data_path, target_workload, samples=[]) -> Context:
        contexts = self.retrieve_contexts(target_data_path, target_workload, samples)
        if len(contexts) == 0:
            raise ValueError(f"No contexts found for target data path: {target_data_path} and workload: {target_workload}")
        
        return contexts[0]
    

    def retrieve_contexts(self, target_data_path, target_workload, samples=[]) -> list[ContextSimilarity]:
        """
        Retrieve contexts based on the target data path and workload.
        :param target_data_path: Path to the target dataset.
        :param target_workload: Workload label for the target dataset.
        :param samples: Optional samples to use for context retrieval.
        :return: List of Context objects.
        """
        target_hardware = os.path.basename(target_data_path)
        if len(samples) < self.min_samples:
            key = (target_hardware, target_workload)
            if key in self.static_similar_context_map:
                return self.static_similar_context_map[key]
            else:
                contexts = self.static_context_retrieval.retrieve_contexts(Context(samples, target_workload, target_hardware))
                self.static_similar_context_map[key] = contexts
                return self.static_similar_context_map[key]
        else:
            contexts = self.dynamic_context_retrieval.retrieve_contexts(Context(samples, target_workload, target_hardware))
            return contexts
