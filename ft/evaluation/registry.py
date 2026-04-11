import importlib
import pkgutil
import inspect
from ft.evaluation.evaluators.base import BaseEvaluator
import ft.evaluation.evaluators


def get_evaluator_registry():
    """
    Dynamically discover and register all evaluators in the evaluation.evaluators package.
    """
    registry = {}
    for _, modname, _ in pkgutil.iter_modules(ft.evaluation.evaluators.__path__):
        module = importlib.import_module(f"ft.evaluation.evaluators.{modname}")
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseEvaluator) and obj is not BaseEvaluator:
                instance = obj()
                registry[instance.name()] = instance

    return registry
