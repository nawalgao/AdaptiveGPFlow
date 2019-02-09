# Imports
import numpy as np

from .function2d import Function2D



# Problem
class Branin(Function2D):
    """ Branin Function. """

    def __init__(self):
        """ Constructor. """
        # Constants
        self.a = 1
        self.b = 5.1/(4*np.pi**2)
        self.c = 5/np.pi
        self.r = 6
        self.s = 10
        self.t = 1/(8*np.pi)
        # Information
        self.min = np.array([[-np.pi, 12.275],
                             [np.pi, 2.275],
                             [9.42478, 2.475]])
        self.value = 0.397887
        self.domain = np.array([[-5, 10], [0, 15]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Branin Function"
        self.latex_tpye = "Other"
        self.latex_cost = "\[ f(\mathbf{x}) = a (x_1 - b x_0^2 + c x_0 - r)^2 + s(1-t) \cos(x_0) + s \]"
        self.latex_desc = "The Branin, or Branin-Hoo, function has three global minima."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = self.a*(x[1] - self.b*x[0]**2 + self.c*x[0] - self.r)**2 + self.s*(1- self.t)*np.cos(x[0]) + self.s
        # Return Cost
        return c
