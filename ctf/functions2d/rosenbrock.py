# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Rosenbrock(Function2D):
    """ Rosenbrock Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([1.0, 1.0])
        self.value = 0.0
        self.domain = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
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
        c = 100.0*(x[1] - x[0]**2.0)**2.0 + (x[0] - 1.0)**2.0
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = -400.0*x[0]*(x[1] - x[0]**2.0) + 2.0*(x[0] - 1.0)
        g[1] = 200.0*(x[1] - x[0]**2.0)
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = -400.0*x[1] + 1200.0*x[0]**2.0 + 2.0
        h[0][1] = -400.0*x[0]
        h[1][0] = h[0][1]
        h[1][1] = 200.0
        # Return Hess
        return h