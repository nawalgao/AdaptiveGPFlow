# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Perm(Function2D):
    """ Perm Function. """

    def __init__(self, n):
        """ Constructor. """
        # Constant
        self.beta = 1
        # Information
        self.min = np.array([1/(i + 1) for i in range(0, n)])
        self.value = 0.0
        self.domain = np.array([[-2, 2] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Perm Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=1}^{d} \left ( \sum_{j=0}^{d-1} (j + 1 + \beta) \left ( x_j^i - \frac{1}{(j +1)^i} \right ) \right )^2 \]"
        self.latex_desc = "..."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([np.sum([(j + 1 + self.beta)*(x[j]**i - 1/((j+1)**i)) for j in range(0, self.n)], axis=0)**2 for i in range(1, self.n+1)], axis=0)
        # Return Cost
        return c
