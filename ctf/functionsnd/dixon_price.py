# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class DixonPrice(Function2D):
    """ Dixon-Price Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([2**(-(2**i - 2)/(2**i)) for i in range(1, 3)])
        self.value = 0.0
        self.domain = np.array([[-10, 10] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Dixon-Price Function"
        self.latex_type = "Valley Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = (x_0 - 1)^2 + \sum_{i=1}^{d-1} i(2x_i^2 - x_{i-1})^2 \]"
        self.latex_desc = " Valley with two local minima."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = (x[0] - 1)**2 + np.sum([(i+1)*(2*x[i]**2 - x[i-1])**2 for i in range(1, self.n - 1)], axis=0)
        # Return Cost
        return c