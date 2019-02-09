# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Trid(Function2D):
    """ Trid Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([2.0, 2.0])
        self.value = 0.0
        self.domain = np.array([[-4, 4], [-4, 4]])
        self.n = 2
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
        c = np.sum([(x[i]-1)**2 for i in range(0, 2)], axis=0) - np.sum([x[i]*x[i-1] for i in range(1, 2)], axis=0)
        # Return Cost
        return c
