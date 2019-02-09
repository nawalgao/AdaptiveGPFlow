# Imports
import numpy as np

from ctf.functions1d.function1d import Function1D



# Problem
class Absolute(Function1D):
    """ Sphere Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0])
        self.value = 0.0
        self.domain = np.array([[-np.inf, np.inf]])
        self.n = 1
        self.smooth = False
        self.info = [True, True, True]
        # Description
        self.latex_name = "Absolute Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(x) = |x| \]"
        self.latex_desc = "It is continuous, convex and unimodal."

    def cost(self, x):
        """ Cost function. """
        # Calculate Cost
        c = np.abs(x[0])
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Calculate Grad
        g = np.sign(x[0])
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Calculate Hess
        h = np.zeros_like(x[0])
        # Return Hess
        return h