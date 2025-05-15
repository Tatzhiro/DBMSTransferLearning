from regression.instance_similarity import InstanceSimilarity

class NearestContext:
    """
    Class to find the nearest context in a list of contexts.
    """

    def __init__(self, instance_similarity: InstanceSimilarity):
        """
        Initialize the NearestContext with InstanceSimilarity.

        :param instance_similarity: An instance of InstanceSimilarity to calculate distances.
        """
        self.instance_similarity = instance_similarity

    def find_nearest(self, target, workload):
        """
        Find the nearest context to the target.

        :param target: The target context to compare against.
        :return: The nearest context from the list.
        """
        datasets: list[InstanceSimilarity.DatasetMetadata] = \
            self.instance_similarity.get_similar_datasets(target, workload, n=1, metadata=True)
        print(f"Using {datasets[0].workload_label}, {datasets[0].hardware_label} as base data")
        return datasets[0]