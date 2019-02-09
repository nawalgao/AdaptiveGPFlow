# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class CrossInTray(Function2D):
    """ Cross in Tray Function. """

    def __init__(self):
        """ Constructor. """
        # Information
        self.min = np.array([[1.34941, 1.34941],
                             [1.34941, -1.34941],
                             [-1.34941, 1.34941],
                             [-1.34941, -1.34941]])
        self.value = np.array([-2.06261, -2.06261, -2.06261, -2.06261])
        self.domain = np.array([[-10.0, 10.0], [-10.0, 10.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Cross in Tray Function"
        self.latex_type = "Many Local Minima"
        self.latex_cost = r"\[ f(\mathbf{x}) = -0.0001 \left( \left| \sin \left(x_0\right) \sin \left(x_1\right) \exp \left( \left|100 - \frac{\sqrt{x_0^{2} + x_1^{2}}}{\pi} \right|\right)\right| + 1 \right)^{0.1} \]"
        self.latex_desc = "The Cross-in-Tray function has multiple global minima."

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = -0.0001*(np.abs(np.sin(x[0])*np.sin(x[1])*np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))) + 1)**0.1
        # Return Cost
        return c

