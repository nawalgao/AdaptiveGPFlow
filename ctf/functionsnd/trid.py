# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Trid(Function2D):
    """ Trid Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([np.nan for i in range(0, n)])
        self.value = np.nan
        self.domain = np.array([[-n**2, n**2] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Trid Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = "\[ f(\mathbf{x}) = \sum_{i=0}^{d-1} (x_i - 1)^2 - \sum_{i=1}^{d-1} x_i x_{i-1} \]"
        self.latex_desc = "The Trid function has no local minimum except the global one. It is shown here in its" \
                          "two-dimensional form. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([(x[i]-1)**2 for i in range(0, self.n)], axis=0) - np.sum([x[i]*x[i-1] for i in range(1, self.n)], axis=0)
        # Return Cost
        return c
