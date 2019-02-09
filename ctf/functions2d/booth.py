# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Booth(Function2D):
    """ Booth's Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([1.0, 3.0])
        self.value = 0.0
        self.domain = np.array([[-10.0, 10.0], [-10.0, 10.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Booth Function"
        self.latex_type = "Plate-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = (x_0 + 2 x_1 - 7)^2 + (2x_0 + x_2 - 5)^2 \]"
        self.latex_desc = "Plate shaped function."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = (x[0] + 2.0*x[1] - 7.0)**2.0 + (2.0*x[0] + x[1] - 5.0)**2.0
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = 2.0*(x[0] + 2.0*x[1] - 7.0) + 4.0*(2.0*x[0] + x[1] - 5.0)
        g[1] = 4.0*(x[0] + 2.0*x[1] - 7.0) + 2.0*(2.0*x[0] + x[1] - 5.0)
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = 10.0
        h[0][1] = 8.0
        h[1][0] = h[0][1]
        h[1][1] = 10.0
        # Return Hess
        return h