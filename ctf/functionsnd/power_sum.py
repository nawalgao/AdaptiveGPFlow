# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class PowerSum(Function2D):
    """ Power Sum Function. """

    def __init__(self, n):
        """ Constructor. """
        # Constants
        self.b = np.array(range(1, n+1))
        # Information
        self.min = np.array([np.nan for i in range(0, n)])
        self.value = 0.0
        self.domain = np.array([[0, 2.0] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Power Sum Function"
        self.latex_type = "Plate-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^{d-1} \left [ \left ( \sum_{j=0}^{d-1} x_j^i \right ) - b_i \right ] \]"
        self.latex_desc = "Multi-dimensional plate-like function. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([(np.sum([x[j]**(i+1) for j in range(0, self.n)], axis=0) - self.b[i])**2 for i in range(0, self.n)], axis=0)
        # Return Cost
        return c
