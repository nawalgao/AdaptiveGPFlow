# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Absolute(Function2D):
    """ Sphere Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([0.0 for i in range(0, n)])
        self.value = 0.0
        self.domain = np.array([[-np.inf, np.inf] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Absolute Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^{d-1} |x_i| \]"
        self.latex_desc = "It is continuous, convex and unimodal."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([np.abs(x) for i in range(0, self.n)])
        # Return Cost
        return c