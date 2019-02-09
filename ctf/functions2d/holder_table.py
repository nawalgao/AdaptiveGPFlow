# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class HolderTable(Function2D):
    """ Holder Table Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([[8.05502, 9.66459],
                             [-8.05502, 9.66459],
                             [8.05502, -9.66459],
                             [-8.05502, -9.66459]])
        self.value = np.array([-19.2085, -19.2085, -19.2085, -19.2085])
        self.domain = np.array([[-10, 10], [-10, 10]])
        self.n = 2
        self.smooth = False
        self.info = [True, False, False]
        # Description
        self.latex_name = "Holder Table Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = \left | \sin(x_0) \cos(x_1) \exp \left ( \left | 1 - \frac{\sqrt{x_0^2 + x_1^2}}{\pi} \right | \right ) \right | \]"
        self.latex_desc = "The Holder Table function has many local minima, with four global minima. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2)/np.pi)))
        # Return Cost
        return c