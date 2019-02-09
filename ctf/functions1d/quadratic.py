# Imports
import numpy as np

from ctf.functions1d.function1d import Function1D



# Problem
class Quadratic(Function1D):
    """ Quadratic Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0])
        self.value = 0.0
        self.domain = np.array([[-np.inf, np.inf]])
        self.n = 1
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Quadratic Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(x) = x^2 \]"
        self.latex_desc = "Simple quadratic function."

    def cost(self, x):
        """ Cost function. """
        # Calculate Cost
        c = x[0]**2.0
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Calculate Grad
        g = 2.0*x[0]
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Calculate Hess
        h = 2.0
        # Return Hess
        return h
