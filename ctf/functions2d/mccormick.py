# Imports
import numpy as np
from numpy import cos, sin

from ctf.functions2d.function2d import Function2D



# Problem
class McCormick(Function2D):
    """ McCormick Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([-0.54719, -1.54719])
        self.value = -1.9133
        self.domain = np.array([[-1.5, 4.0], [-3.0, 4.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "McCormick Function"
        self.latex_type = "Plate-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sin(x_0 + x_1) + (x_0 - x_1)^2 - 1.5 x_0 + 2.5 x_1 + 1 \]"
        self.latex_desc = "description"

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -1.5*x[0] + 2.5*x[1] + (x[0] - x[1])**2 + sin(x[0] + x[1]) + 1
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = 2*x[0] - 2*x[1] + cos(x[0] + x[1]) - 1.5
        g[1] = -2*x[0] + 2*x[1] + cos(x[0] + x[1]) + 2.5
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = -sin(x[0] + x[1]) + 2
        h[0][1] = -sin(x[0] + x[1]) - 2
        h[1][0] = h[0][1]
        h[1][1] = -sin(x[0] + x[1]) + 2
        # Return Hess
        return h