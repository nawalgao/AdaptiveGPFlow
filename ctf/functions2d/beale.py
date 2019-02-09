# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Beale(Function2D):
    """ Beale's Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([3.0, 0.5])
        self.value = 0.0
        self.domain = np.array([[-4.5, 4.5], [-4.5, 4.5]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Beale's Function"
        self.latex_tpye = "Other"
        self.latex_cost = r"\[ f(\mathbf{x}) = (1.5 - x_0 + x_0 x_1)^2 + (2.25 - x_0 + x_0 x_1^2)^2 + (2.625 - x_0 + x_0 x_1^3)^2 \]"
        self.latex_desc = "The Beale function is multimodal, with sharp peaks at the corners of the input domain. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = (1.5 - x[0] + x[0]*x[1])**2.0 + (2.25 - x[0] + x[0]*x[1]**2.0)**2.0 + (2.625 - x[0] + x[0]*x[1]**3.0)**2.0
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = (2*x[1] - 2)*(x[0]*x[1] - x[0] + 1.5) + (2*x[1]**2 - 2)*(x[0]*x[1]**2 - x[0] + 2.25) + (2*x[1]**3 - 2)*(x[0]*x[1]**3 - x[0] + 2.625)
        g[1] = 6*x[0]*x[1]**2*(x[0]*x[1]**3 - x[0] + 2.625) + 4*x[0]*x[1]*(x[0]*x[1]**2 - x[0] + 2.25) + 2*x[0]*(x[0]*x[1] - x[0] + 1.5)
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = (x[1] - 1)*(2*x[1] - 2) + (x[1]**2 - 1)*(2*x[1]**2 - 2) + (x[1]**3 - 1)*(2*x[1]**3 - 2)
        h[0][1] = 6*x[0]*x[1]**2*(x[1]**3 - 1) + 4*x[0]*x[1]*(x[1]**2 - 1) + 2*x[0]*x[1] + 2*x[0]*(x[1] - 1) - 2*x[0] + 6*x[1]**2*(x[0]*x[1]**3 - x[0] + 2.625) + 4*x[1]*(x[0]*x[1]**2 - x[0] + 2.25) + 3.0
        h[1][0] = h[0][1]
        h[1][1] = 18*x[0]**2*x[1]**4 + 8*x[0]**2*x[1]**2 + 2*x[0]**2 + 12*x[0]*x[1]*(x[0]*x[1]**3 - x[0] + 2.625) + 4*x[0]*(x[0]*x[1]**2 - x[0] + 2.25)
        # Return Hess
        return h