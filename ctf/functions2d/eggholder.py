# Imports
import numpy as np
from numpy import sin, sqrt

from ctf.functions2d.function2d import Function2D



# Problem
class Eggholder(Function2D):
    """ Eggholder Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([512, 404.2319])
        self.value = -959.6407
        self.domain = np.array([[-512.0, 512.0], [-512.0, 512.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Eggholder Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = - \left(x_1+47\right) \sin \left(\sqrt{\left|x_1 + \frac{x}{2}+47\right|}\right) - x_0 \sin \left(\sqrt{\left|x_0 - \left(x_1 + 47 \right)\right|}\right) \]"
        self.latex_desc = "The Eggholder function is a difficult function to optimize, because of the large number of local minima. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -x[0]*sin(sqrt(abs(x[0] - x[1] - 47))) + (-x[1] - 47)*sin(sqrt(abs(x[0]/2 + x[1] + 47)))
        # Return Cost
        return c
