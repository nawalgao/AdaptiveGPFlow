# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class DropWave(Function2D):
    """ Matyas Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = np.array([-1.0])
        self.domain = np.array([[-5.12, 5.12], [-5.12, 5.12]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Drop Wave Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = - \frac{1 + \cos(12\sqrt{x_1^2 + x_2^2})}{0.5(x_0^2 + x_1^2) + 2} \]"
        self.latex_desc = "The Drop-Wave function is multimodal and highly complex."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -(1 + np.cos(12*np.sqrt(x[0]**2 + x[1]**2)))/(0.5*(x[0]**2 + x[1]**2) + 2)
        # Return Cost
        return c

