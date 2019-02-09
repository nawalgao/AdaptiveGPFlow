# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Absolute(Function2D):
    """ Sphere Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Absolute Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^{d-1} |x_i| \]"
        self.latex_desc = "It is continuous, convex and unimodal."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.abs(x[0]) + np.abs(x[1])
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = np.sign(x[0])
        g[1] = np.sign(x[1])
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = 0.0
        h[0][1] = 0.0
        h[0][1] = h[1][0]
        h[1][1] = 0.0
        # Return Hess
        return h