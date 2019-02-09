# Imports
import numpy as np
from numpy import exp, cos, sin, pi

from ctf.functions2d.function2d import Function2D



# Problem
class Easom(Function2D):
    """ Easom Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([pi, pi])
        self.value = -1
        self.domain = np.array([[-100.0, 100.0], [-100.0, 100.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Easom Function"
        self.latex_cost = r"\[ f(\mathbf{x}) = - \cos(x_0) \cos(x_1) \exp(-(x_0 - \pi)^2-(x_1 - \pi)^2) \]"
        self.latex_desc = "The Easom function has several local minima. It is unimodal, and the global minimum has a " \
                          "small area relative to the search space. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1])
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = -(-2*x[0] + 2*pi)*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1]) + exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*sin(x[0])*cos(x[1])
        g[1] = -(-2*x[1] + 2*pi)*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1]) + exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*sin(x[1])*cos(x[0])
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = -(-2*x[0] + 2*pi)**2*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1]) + 2*(-2*x[0] + 2*pi)*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*sin(x[0])*cos(x[1]) + 3*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1])
        h[0][1] = -(-2*x[0] + 2*pi)*(-2*x[1] + 2*pi)*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1]) + (-2*x[0] + 2*pi)*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*sin(x[1])*cos(x[0]) + (-2*x[1] + 2*pi)*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*sin(x[0])*cos(x[1]) - exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*sin(x[0])*sin(x[1])
        h[1][0] = h[0][1]
        h[1][1] = -(-2*x[1] + 2*pi)**2*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1]) + 2*(-2*x[1] + 2*pi)*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*sin(x[1])*cos(x[0]) + 3*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)*cos(x[0])*cos(x[1])
        # Return Hess
        return h