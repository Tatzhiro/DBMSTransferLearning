from abc import ABC, abstractmethod

from regression.context_retrieval.context_retrieval import Context, ContextSimilarity


class DynamicContextRetrieval(ABC):
    """
    Abstract base class for dynamic context retrieval methods.
    """
    @abstractmethod
    def retrieve_contexts(self, target_context: Context) -> list[ContextSimilarity]:
        """
        Retrieve contexts based on the provided samples.
        :param target_context: Context object containing the target samples, workload, and hardware information.
        :return: Context object containing the retrieved context information.
        """
        pass
