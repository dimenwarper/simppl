# simppl
#### The simple (?) probabilistic programming language™

`simppl` is not your typical probabilistic programming language. 
It does not include the latest and greatest inference methods.
In fact, it only has exhaustive/enumerative inference.
It does not have your favorite distribution; it only includes a couple of very simplified, discretized distribution. 
It doesn't use that new awesome computational graph/deep learning framework, it has minimal dependencies beyond the python data stack.
Because of this, models will be dead simple and inference should hopefully be fast and straightforward. 
Rather than a library you should use for production, it is a sandbox for quirky Bayesian inference ideas and for starting conversations around the theme of "Bayesian inference for users with deadlines" (with apologies to Django).
Read the blogpost [here](http://hyperparameter.com/).

## Install

With pip:

```

```

## A simppl model

A simple model illustrating that `simppl` flavor:

```python
from simppl.distributions import SomeValue
from simppl.utils import capture_locals
from simppl.inference import Exhaustive

import numpy as np

def sigmoid_response(toxicity, log_concentration):
    return 1 / (1 + np.exp(toxicity * log_concentration))

log_concentration = 1
true_toxicity = 0.6
inhibitions = (sigmoid_response(true_toxicity, log_concentration) + np.random.randn(2) * 0.001).reshape(-1, 1)

def cytotoxicity_model(inibitions=inhibitions):
    toxicity = SomeValue('toxicity', between=[0, 2], around=[0, 1.5], mostly=1.3)
    inhibition = SomeValue(
        'inibition', 
        between=[0, 1], 
        mostly=sigmoid_response(toxicity, log_concentration),
        observations=inhibitions
    )
                        
    capture_locals()
    return inhibition

env = Exhaustive(cytotoxicity_model)
(env
 .executions
 .assign(toxicity=lambda df: np.round(df['toxicity'], 2))
 .groupby('toxicity')['_probability_']
 .sum()
 .plot.bar()
);
```

This and other examples are in the showcase.ipynb.

  
️
