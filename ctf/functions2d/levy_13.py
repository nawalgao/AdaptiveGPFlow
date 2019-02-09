# Imports
import numpy as np
from numpy import cos, sin, pi

from ctf.functions2d.function2d import Function2D



# Problem
class Levy13(Function2D):
    """ Levy 13 Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([1.0, 1.0])
        self.value = 0.0
        self.domain = np.array([[-10.0, 10.0], [-10.0, 10.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Levy No. 13 Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sin^2(3 \pi x_0) + (x_0 - 1)^2 [1 + \sin^2(3 \pi x_1)] + (x_1 - 1)^2 [1 + \sin^2(2 \pi x_1)] \]"
        self.latex_desc = "description"

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = (x[0] - 1)**2*(sin(3*pi*x[1])**2 + 1) + (x[1] - 1)**2*(sin(2*pi*x[1])**2 + 1) + sin(3*pi*x[0])**2
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = (2*x[0] - 2)*(sin(3*pi*x[1])**2 + 1) + 6*pi*sin(3*pi*x[0])*cos(3*pi*x[0])
        g[1] = (2*x[0] - 2)*(sin(3*pi*x[1])**2 + 1) + 6*pi*sin(3*pi*x[0])*cos(3*pi*x[0])
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate  Hess
        h[0][0] = -18*pi**2*sin(3*pi*x[0])**2 + 2*sin(3*pi*x[1])**2 + 18*pi**2*cos(3*pi*x[0])**2 + 2
        h[0][1] = 6*pi*(2*x[0] - 2)*sin(3*pi*x[1])*cos(3*pi*x[1])
        h[1][0] = h[0][1]
        h[1][1] = 6*pi*(2*x[0] - 2)*sin(3*pi*x[1])*cos(3*pi*x[1])
        # Return Grad
        return h