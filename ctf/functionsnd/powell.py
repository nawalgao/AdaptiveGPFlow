# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Powell(Function2D):
    """ Powell Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-4, 5], [-4, 5]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Powell Function"
        self.latex_tpye = "Other"
        self.latex_cost = "\[ f(x,y) = ... \]"
        self.latex_desc = "... "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c =
        # Return Cost
        return c
