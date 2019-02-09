# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Bukin6(Function2D):
    """ Bukin No. 6 Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([-10.0, 1.0])
        self.value = 0.0
        self.domain = np.array([[-15.0, -5.0], [-3.0, 3.0]])
        self.n = 2
        self.smooth = False
        self.info = [True, False, False]
        # Description
        self.latex_name = "Bukin Function No.6"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = 100\sqrt{\left|x_1 - 0.01x_0^{2}\right|} + 0.01 \left|x_0+10 \right| \]"
        self.latex_desc = "The sixth Bukin function has many local minima, all of which lie in a ridge.  "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = 100.0*np.sqrt(np.abs(x[1] - 0.01*x[0]**2)) + 0.01*np.abs(x[0] + 10.0)
        # Return Cost
        return c