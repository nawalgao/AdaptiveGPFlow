# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class StyblinskiTang(Function2D):
    """ Styblinski-Tang Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([-2.903534 for i in range(0, n)])
        self.value = -39.16599*n
        self.domain = np.array([[-5.0, 5.0] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Styblinski-Tang Function"
        self.latex_type = "Other"
        self.latex_cost = r"\[ f(\mathbf{x}) = \frac{1}{2} \sum_{i=0}^{d-1} (x_i^4 - 16 x_i^2 + 5 x_i) \]"
        self.latex_desc = "The local minima are separated by a local maximum. There is only a single global minimum."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = 0.5*np.sum([x[i]**4 - 16*x[i]**2 +5*x[i] for i in range(0, self.n)])
        # Return Cost
        return c
