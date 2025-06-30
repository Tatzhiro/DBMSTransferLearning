from abc import ABC, abstractmethod

from regression.context_retrieval.context_retrieval import Context, ContextSimilarity


class StaticContextRetrieval(ABC):
    """
    Abstract base class for Context Retrieval methods.
    """
    @abstractmethod
    def retrieve_contexts(self, target_context: Context) -> list[ContextSimilarity]:
        """
        Get similar datasets based on the provided parameters.
        
        :param target_context: Context object containing the target workload and hardware information.
        :return: Context object containing the similar dataset information.
        """
        pass
