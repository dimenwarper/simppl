from typing import Callable, Optional

from .computation_registry import COMPUTATION_REGISTRY, CNode, wrap_op
import inspect

def capture_locals():
    frame = inspect.currentframe()
    try:
        vars_to_capture = frame.f_back.f_locals
        COMPUTATION_REGISTRY.add_to_model_locals(**vars_to_capture)
    finally:
        del frame

def register_function(att_name: str, fun: Callable, cls: Optional[type]=None) -> Callable:
    setattr(CNode, att_name, wrap_op(att_name, lambda x, fun_name: getattr(x, fun_name)()))
    if cls is not None:
        setattr(cls, att_name, lambda x: fun(x))
