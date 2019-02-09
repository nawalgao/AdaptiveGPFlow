# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class ThreeHumpCamel(Function2D):
    """ Three Hump Camel Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
        # Description
        self.latex_name = "Three Hump Camel Function"
        self.latex_type = "Valley Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = 2x_0^{2} - 1.05x_0^{4} + \frac{x_0^{6}}{6} + x_0 x_1 +  ^{2} \]"
        self.latex_desc = "The function has three local minima.  "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = 2.0*x[0]**2.0 - 1.05*x[0]**4.0 + (x[0]**6.0)/6.0 + x[0]*x[1] + x[1]**2.0
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = 4.0*x[0]**1.0 - 4.2*x[0]**3.0 + 1.0*x[0]**5.0 + x[1]
        g[1] = x[0] + 2.0*x[1]**1.0
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = -12.6*x[0]**2.0 + 5.0*x[0]**4.0 + 4.0
        h[0][1] = 1.0
        h[1][0] = h[0][1]
        h[1][1] = 2.0
        # Return Hess
        return h