# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Rosenbrock(Function2D):
    """ Rosenbrock Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([1.0 for i in range(0, n)])
        self.value = 0.0
        self.domain = np.array([[-np.inf, np.inf] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Rosenbrock Function"
        self.latex_type = "Valley Shaped"
        self.latex_cost = r"\[ f(\boldsymbol{x}) = \sum_{i=0}^{d-2} \left[ 100 \left(x_{i+1} - x_{i}^{2}\right)^{2} + \left(x_{i} - 1\right)^{2}\right] \]"
        self.latex_desc = "The Rosenbrock function, also referred to as the Valley or Banana function, is a popular " \
                          "test problem for gradient-based optimization algorithms. It is shown in the plot above in " \
                          "its two-dimensional form. The function is unimodal, and the global minimum lies in a " \
                          "narrow, parabolic valley. However, even though this valley is easy to find, convergence " \
                          "to the minimum is difficult."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(0, self.n-1)])
        # Return Cost
        return c
