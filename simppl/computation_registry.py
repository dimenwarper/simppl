from copy import deepcopy
from typing import Union, Callable, Dict, Any, List
import numpy as np


class ComputationRegistry:
    def __init__(self):
        self.current_definitions = {}
        self.current_variables = {}

        self.model_locals = {}

        self.resetting = False

    def add_to_model_locals(self, **kwargs):
        if not self.resetting:
            for k, v in kwargs.items():
                if k not in self.model_locals:
                    self.model_locals[k] = []
                self.model_locals[k].append([v])

    def reset(self, fun, **fun_kwargs):
        self.current_definitions = {}
        self.current_variables = {}
        self.model_locals = {}
        self.resetting = True
        fun(**fun_kwargs)
        self.resetting = False

    def call_or_create_variable(self, name, cls, register, *args, **kwargs):
        if name not in self.current_variables or not register:
            instance = super(Variable, cls).__new__(cls)
            instance.name = name
            instance.__init__(name, *args, **kwargs)

            if register:
                self.current_variables[name] = instance
            return instance

        if len(self.current_definitions) > 0:
            return self.current_variables[name].call(self.current_definitions)

    def call_with_definitions(self, fun: Callable, definitions: Dict[str, Any]):
        self.current_definitions = definitions
        variable_clones = {}
        for name, var in self.current_variables.items():
            clone = type(var)(name, register=False)
            for att, val in var.__dict__.items():
                if isinstance(val, CNode):
                    setattr(clone, att, val.call(definitions))
                else:
                    setattr(clone, att, val)
            variable_clones[name] = clone

        score = sum([variable_clones[name].score(val) for name, val in definitions.items()])
        res = fun()
        if len(self.model_locals) > 0:
            for k, vals in self.model_locals.items():
                vals[-1].append(score)
        return res, score


COMPUTATION_REGISTRY = ComputationRegistry()


class CNode:
    def call(self, definitions: Dict[str, Any]):
        raise NotImplementedError('CNode.call should be implemented')


class Constant(CNode):
    def __init__(self, value):
        self.value = value

    def call(self, definitions: Dict[str, Any]):
        return self.value

    def __repr__(self):
        return f'Constant(value={self.value})'


class Variable(CNode):
    def __new__(cls, name: str, *args, register: bool = True, **kwargs):
        return COMPUTATION_REGISTRY.call_or_create_variable(name, cls, register, *args, **kwargs)

    def call(self, definitions: Dict[str, Any]):
        return definitions[self.name]


class Op(CNode):
    def __init__(self, name: str, fun_name: str, fun: Callable, args: List[CNode]):
        self.name = name
        self.fun_name = fun_name
        self.fun = fun
        self.args = args

    def call(self, definitions: Dict[str, Any]):
        call_args = [arg.call(definitions) for arg in self.args]
        return self.fun(*call_args, self.fun_name)

    def __repr__(self):
        return f'Op(name={self.name}, fun={self.fun}, args={self.args})'


__OPS_TO_ADD = [
    '__add__',
    '__mul__',
    '__sub__',
    '__truediv__',
    '__pow__',
    '__or__',
    '__le__',
    '__ge__',
    '__ne__',
    '__eq__',
    '__lt__',
    '__gt__',
    '__radd__',
    '__rmul__',
    '__rsub__',
    '__rtruediv__',
    '__rpow__',
    '__ror__',
    '__rand__',
    '__rle__',
    '__rge__',
    '__rlt__',
    '__rgt__',
    'dot',
    'matmul',
    'log',
    'exp',
]

__SPECIAL_OPS = {
    'dot': lambda x, y, fun_name: np.dot(x, y),
    'matmul': lambda x, y, fun_name: np.matmul(x, y),
    'log': lambda x, y, fun_name: np.log(y),
    'exp': lambda x, y, fun_name: np.exp(y),
}


def wrap_op(fun_name: str, fun: Callable) -> Callable:
    def wrapped(self, *args):
        name = f'{fun_name}({self.name})'
        op_args = [self] + [v if isinstance(v, CNode) else Constant(value=v) for v in args]
        return Op(name=name, fun_name=fun_name, fun=fun, args=op_args)

    return wrapped


for att_name in __OPS_TO_ADD:
    if att_name in __SPECIAL_OPS:
        fun = __SPECIAL_OPS[att_name]
    else:
        fun = lambda x, y, fun_name: getattr(x, fun_name)(y)
    setattr(CNode, att_name, wrap_op(att_name, fun))