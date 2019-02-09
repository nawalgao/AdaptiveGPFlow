# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Schaffer4(Function2D):
    """ Schaffer No. 4 Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-100.0, 100.0], [-100.0, 100.0]])
        self.n = 2
        self.smooth = False
        self.info = [True, False, False]
        # Description
        self.latex_name = "Schaffer No. 4 Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = 0.5 + \frac{\cos(\sin(|x_0^2 - x_1^2|)) - 0.5}{[1 + 0.001(x_0^2 + x_1^2)]^2} \]"
        self.latex_desc = "The fourth Schaffer function has many local minima. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = 0.5 + (np.cos(np.sin(np.abs(x[0]**2 - x[1]**2))) - 0.5)/(1 + 0.001*(x[0]**2 + x[1]**2))**2
        # Return Cost
        return c