# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class SumSquares(Function2D):
    """ Sum Squares Function. """

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
        self.latex_name = "Sum Squares Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^d  (i + 1) x_i^2 \]"
        self.latex_desc = "The Sum Squares function, also referred to as the Axis Parallel Hyper-Ellipsoid" \
                          "function, has no local minimum except the global one. It is continuous, convex and unimodal."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([i*x[i]**2 for i in range(0, 2)], axis=0)
        # Return Cost
        return c
