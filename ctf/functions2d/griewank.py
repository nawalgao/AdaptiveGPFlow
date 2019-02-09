# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Griewank(Function2D):
    """ Griewank Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-600.0, 600.0], [-600.0, 600.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Griewank Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^{d-1} \frac{x_i^2}{4000} - \prod^{i=0}^{d-1} \cos \left ( \frac{x_i}{\sqrt{i}} \right ) + 1 \]"
        self.latex_desc = "The Griewank function has many widespread local minima, which are regularly distributed."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = (x[0]**2 + x[1]**2)/4000 - np.cos(x[0])*np.cos(x[1]/2) + 1
        # Return Cost
        return c
