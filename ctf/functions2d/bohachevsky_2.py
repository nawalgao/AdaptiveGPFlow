# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Bohachevsky2(Function2D):
    """ Bohachevsky No. 2 Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Bohachevsky No. 2 Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = x_0^2 + 2x_1^2 - 0.3 \cos(3 \pi x_0) \cos(4 \pi x_1) + 0.3 \]"
        self.latex_desc = "A bowl-shaped function."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0])*np.cos(4*np.pi*x[1]) + 0.3
        # Return Cost
        return c
