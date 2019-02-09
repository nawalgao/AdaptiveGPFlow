# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class StyblinskiTang(Function2D):
    """ Styblinski-Tang Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([-2.903534, -2.903534])
        self.value = -39.16599*2.0
        self.domain = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Styblinski-Tang Function"
        self.latex_type = "Other"
        self.latex_cost = r'\[ f(\mathbf{x}) = \frac{1}{2} \sum_{i=0}^{d-1} (x_i^4 - 16 x_i^2 + 5 x_i) \]'
        self.latex_desc = "The local minima are separated by a local maximum. There is only a single global minimum."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = 0.5*(x[0]**4.0 - 16*x[0]**2.0 + 5.0*x[0] + x[1]**4.0 - 16*x[1]**2.0 + 5.0*x[1])
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = -16.0*x[0]**1.0 + 2.0*x[0]**3.0 + 2.5
        g[1] = -16.0*x[1]**1.0 + 2.0*x[1]**3.0 + 2.5
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = 6.0*x[0]**2.0 - 16.0
        h[0][1] = 0.0
        h[1][0] = h[0][1]
        h[1][1] = 6.0*x[1]**2.0 - 16.0
        # Return Hess
        return h