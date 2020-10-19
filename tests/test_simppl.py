import unittest
from simppl.distributions import Pick, Flip
from simppl.inference import Exhaustive
from simppl.utils import capture_locals
import numpy as np

class TestModels(unittest.TestCase):
    def test_basic_model(self):
        # All observations are 2D arrays, rows are samples, features are columns
        tosses = np.array([0, 0, 0, 1, 0, 0]).reshape(-1, 1)

        def coinflip_model(tosses=tosses):
            p = Pick('flip_probas', items=[0.1, 0.5, 0.8, 0.9])
            coinflip = Flip('coinflip', p=p, observations=tosses)

            capture_locals()  # --> capture_locals grabs the state of all variables above it and records ,
            # them at each execution...these values can then be inspected after inference...
            return coinflip

        # Only one inference method available for now: Exhaustive for enumerating all possibilities
        env = Exhaustive(coinflip_model)

        p_probs = (env
         .executions
         .groupby('p')['_probability_']
         .sum()
         )
        self.assertEqual(p_probs.index[p_probs.argmax()], 0.1)

if __name__ == '__main__':
    unittest.main()