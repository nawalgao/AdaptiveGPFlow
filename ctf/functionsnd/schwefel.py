# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Schwefel(Function2D):
    """ Schwefel Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([420.9687 for i in range(0, n)])
        self.value = 0.0
        self.domain = np.array([[-500, 500] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Schwefel Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = 418.9829 d - \sum_{i=0}^{d-1} x_i \sin(\sqrt{|x_i|}) \]"
        self.latex_desc = "The Schwefel function is complex, with many local minima."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = 418.9829*2 - np.sum([x[i]*np.sin(np.sqrt(np.abs(x[i]))) for i in range(0, self.n)])
        # Return Cost
        return c