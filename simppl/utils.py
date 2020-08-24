from .computation_registry import COMPUTATION_REGISTRY
import inspect

def capture_locals():
    frame = inspect.currentframe()
    try:
        COMPUTATION_REGISTRY.add_to_model_locals(**frame.f_back.f_locals)
    finally:
        del frame