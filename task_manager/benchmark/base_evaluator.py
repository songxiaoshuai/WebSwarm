"""Minimal interface base class for all benchmark evaluators.

Each benchmark only needs to implement reset and evaluate. TaskManager depends solely
on these uniform interfaces and does not need to know the dataset's internal format.
"""


class BaseEvaluator:
    """Evaluator base class defining the evaluation interface and common methods."""

    def __init__(self, config: dict = None):
        """Maintain a uniform constructor signature; subclasses interpret config themselves."""
        pass

    def reset(self, task_id=None):
        """Reset the current sample and return the task information visible to the agent."""
        raise NotImplementedError

    def evaluate(self, prediction: str):
        """Evaluate a prediction and return (reward, reward_info)."""
        raise NotImplementedError
