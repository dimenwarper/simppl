class EffectRegistry:
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
        self.resetting = True
        fun(**fun_kwargs)
        self.resetting = False

    def call_variable(self, name, cls, register, *args, **kwargs):
        from .distributions import Distribution
        if name not in self.current_variables or not register:
            instance = super(Distribution, cls).__new__(cls)
            instance.name = name
            instance.__init__(name, *args, **kwargs)

            if register:
                self.current_variables[name] = instance
            return instance

        variable = self.current_variables[name]

        if len(self.current_definitions) > 0:
            return self.current_definitions[name]

    def get_variables(self):
        return self.current_variables

    def call_with_definitions(self, fun, definitions):
        from .distributions import Distribution
        self.current_definitions = definitions
        variable_clones = {}
        for name, var in self.current_variables.items():
            clone = type(var)(name, register=False)
            for att, val in var.__dict__.items():
                if isinstance(val, Distribution):
                    setattr(clone, att, definitions[val.name])
                else:
                    setattr(clone, att, val)
            variable_clones[name] = clone

        score = sum([variable_clones[name].score(val) for name, val in definitions.items()])
        res = fun()
        if len(self.model_locals) > 0:
            for k, vals in self.model_locals.items():
                vals[-1].append(score)
        return res, score

REGISTRY = EffectRegistry()