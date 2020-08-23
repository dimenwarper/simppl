from .registry import REGISTRY
import inspect

def capture_locals():
    frame = inspect.currentframe()
    try:
        REGISTRY.add_to_model_locals(**frame.f_back.f_locals)
    finally:
        del frame