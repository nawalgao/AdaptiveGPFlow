# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class SumOfDifferentPowers(Function2D):
    """ Sum of Different Powers Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-1, 1], [-1, 1]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Sum of Different Powers Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^d  |x_i|^{i+2} \]"
        self.latex_desc = " The Sum of Different Powers function is unimodal. It is shown here in its two-dimensional" \
                          " form.  "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([np.abs(x[i])**(i+2) for i in range(0, 2)], axis=0)
        # Return Cost
        return c
