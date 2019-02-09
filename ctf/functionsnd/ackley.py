# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class Ackley(Function2D):
    """ Ackley's Function. """

    def __init__(self):
        """ Constructor. """
        # Constants
        self.a = 20
        self.b = 0.2
        self.c = 2*np.pi
        # Information
        self.min = np.array([0.0 for i in range(0, self.n)])
        self.value = 0.0
        self.domain = np.array([[-32.768, 32.768] for i in range(0, self.n)])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Ackley's Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x})  = -20\exp\left(-0.2\sqrt{0.5\left(x_0^{2}+x_1^{2}\right)}\right) -\exp\left(0.5\left(\cos\left(2\pi x_0\right)+\cos\left(2\pi x_1\right)\right)\right) + 20 + \exp(1) \]"
        self.latex_desc = "It is characterized by a nearly flat outer region, and a large hole at the centre. The " \
                          "function poses a risk for optimization algorithms, particularly hillclimbing algorithms, " \
                          "to be trapped in one of its many local minima. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -self.a*np.exp(-self.b*np.sqrt(np.sum([x[i]**2 for i in range(0, self.n)])/self.n)) - np.exp(np.sum([np.cos(self.c*x[i]) for i in range(0, self.n)])/self.n) + self.a + np.exp(1)
        # Return Cost
        return c