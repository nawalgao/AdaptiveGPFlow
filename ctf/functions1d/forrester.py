# Imports
import numpy as np

from ctf.functions1d.function1d import Function1D



# Problem
class Forrester(Function1D):
    """ Forrester Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([np.nan])
        self.value = np.nan
        self.domain = np.array([[0.0, 1.0]])
        self.n = 1
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Forrester Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(x) = (6x - 2)^2 \sin(12x - 4) \]"
        self.latex_desc = "This is a simple one-dimensional test function. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sin(12*x[0] - 4)*(6*x[0] - 2)**2
        # Return Cost
        return c
