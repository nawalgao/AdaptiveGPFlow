# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Rastrigin(Function2D):
    """ Rastrigin Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([0.0 for i in range(0, n)])
        self.value = 0.0
        self.domain = np.array([[-5.12, 5.12] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Rastrigin Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = 10d + \sum_{i=0}^{d-1} [x_i^2 - 10 \cos(2 \pi x_i)] \]"
        self.latex_desc = "The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima are regularly distributed."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = 10*self.n + np.sum([x[i]**2 - 10*np.cos(2*np.pi*x[i]) for i in range(0, self.n)])
        # Return Cost
        return c