# Imports
import numpy as np
from numpy import exp, cos, sin, sqrt, pi

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
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        self.n = 2
        self.smooth = True
        self.info = [True, True, True]
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
        c = -20.0*exp(-0.2*sqrt(0.5*(x[0]**2 + x[1]**2))) - exp(0.5*(cos(2.0*pi*x[0]) + cos(2.0*pi*x[1]))) + 20.0 + exp(1.0)
        # Return Cost
        return c

    def grad(self, x):
        """ Grad function. """
        # Grad
        g = np.zeros(x.shape)
        # Calculate Grads
        g[0] = 2.0*x[0]*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/sqrt(0.5*x[0]**2 + 0.5*x[1]**2) + 1.0*pi*exp(0.5*cos(2.0*pi*x[0]) + 0.5*cos(2.0*pi*x[1]))*sin(2.0*pi*x[0])
        g[1] = 2.0*x[1]*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/sqrt(0.5*x[0]**2 + 0.5*x[1]**2) + 1.0*pi*exp(0.5*cos(2.0*pi*x[0]) + 0.5*cos(2.0*pi*x[1]))*sin(2.0*pi*x[1])
        # Return Grad
        return g

    def hess(self, x):
        """ Hess function. """
        # Hess
        h = np.zeros((2, 2) + x.shape[1:])
        # Calculate Hess
        h[0][0] = -0.2*x[0]**2*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/(0.5*x[0]**2 + 0.5*x[1]**2) - 1.0*x[0]**2*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/(0.5*x[0]**2 + 0.5*x[1]**2)**(3/2) - 1.0*np.pi**2*exp(0.5*cos(2.0*np.pi*x[0]) + 0.5*cos(2.0*np.pi*x[1]))*sin(2.0*np.pi*x[0])**2 + 2.0*np.pi**2*exp(0.5*cos(2.0*np.pi*x[0]) + 0.5*cos(2.0*np.pi*x[1]))*cos(2.0*np.pi*x[0]) + 2.0*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/sqrt(0.5*x[0]**2 + 0.5*x[1]**2)
        h[0][1] = -0.2*x[0]*x[1]*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/(0.5*x[0]**2 + 0.5*x[1]**2) - 1.0*x[0]*x[1]*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/(0.5*x[0]**2 + 0.5*x[1]**2)**(3/2) - 1.0*np.pi**2*exp(0.5*cos(2.0*np.pi*x[0]) + 0.5*cos(2.0*np.pi*x[1]))*sin(2.0*np.pi*x[0])*sin(2.0*np.pi*x[1])
        h[0][1] = h[1][0]
        h[1][1] = -0.2*x[1]**2*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/(0.5*x[0]**2 + 0.5*x[1]**2) - 1.0*x[1]**2*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/(0.5*x[0]**2 + 0.5*x[1]**2)**(3/2) - 1.0*np.pi**2*exp(0.5*cos(2.0*np.pi*x[0]) + 0.5*cos(2.0*np.pi*x[1]))*sin(2.0*np.pi*x[1])**2 + 2.0*np.pi**2*exp(0.5*cos(2.0*np.pi*x[0]) + 0.5*cos(2.0*np.pi*x[1]))*cos(2.0*np.pi*x[1]) + 2.0*exp(-0.2*sqrt(0.5*x[0]**2 + 0.5*x[1]**2))/sqrt(0.5*x[0]**2 + 0.5*x[1]**2)
        # Return Hess
        return h