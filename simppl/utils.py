from .computation_registry import COMPUTATION_REGISTRY
import inspect

def capture_locals():
    frame = inspect.currentframe()
    try:
        vars_to_capture = frame.f_back.f_locals
        COMPUTATION_REGISTRY.add_to_model_locals(**vars_to_capture)
    finally:
        del frame