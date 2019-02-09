# Imports
import numpy as np

from ctf.functions1d.function1d import Function1D



# Problem
class GramacyLee(Function1D):
    """ Quadratic Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.55])
        self.value = -0.86808465909
        self.domain = np.array([[0.5, 2.5]])
        self.n = 1
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Gramacy and Lee Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = \frac{\sin(10 \pi x)}{2x} + (x-1)^4 \]"
        self.latex_desc = "This is a simple one-dimensional test function. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sin(10*np.pi*x[0])/(2*x[0]) + (x[0] - 1)**4
        # Return Cost
        return c
