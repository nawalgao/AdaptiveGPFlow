# Imports
import numpy as np
from numpy import sin

from ctf.functions2d.function2d import Function2D



# Problem
class Michalewicz(Function2D):
    """ Michalewicz Function. """

    def __init__(self, n):
        """ Constructor. """
        # Constants
        self.m = 10
        # Information
        self.min = np.array([[np.nan, np.nan] for i in range(n)])
        self.value = np.nan
        self.domain = np.array([[0, np.pi] for i in range(n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Michalewicz"
        self.latex_type = "Steep"
        self.latex_cost = r"\[  f(\mathbf{x}) = \sum_{i=0}^{d-1} \sin(x_i) \sin^{2m} \left ( \frac{(i+1) x_i^2}{\pi} \right ) \]"
        self.latex_desc = "The Michalewicz function is multimodal and has d! local minima. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -np.sum([np.sin(x[i])*sin(((i+1)*x[i]**2)/np.pi)**(2*self.m) for i in range(0, self.n)], axis=0)
        # Return Cost
        return c