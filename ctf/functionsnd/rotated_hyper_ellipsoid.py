# Imports
import numpy as np

from ctf.functions2d.function2d import Function2D



# Problem
class RotatedHyperEllipsoid(Function2D):
    """ Rotated Hyper-Ellipsoid Function. """

    def __init__(self, n):
        """ Constructor. """
        # Information
        self.min = np.array([0.0 for i in range(0, n)])
        self.value = 0.0
        self.domain = np.array([[-65.536, 65.536] for i in range(0, n)])
        self.n = n
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Rotated Hyper-Ellipsoid Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^{d-1} \sum_{j=1}^i x_j^2 \]"
        self.latex_desc = "The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal. It is an " \
                          "extension of the Axis Parallel Hyper-Ellipsoid function, also referred to as the Sum " \
                          "Squares function. The plot shows its two-dimensional form. "

    def cost(self, x):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        # Calculate Cost
        c = np.sum([np.sum([x[j]**2 for j in range(0, i+1)], axis=0) for i in range(0, self.n)], axis=0)
        # Return Cost
        return c
