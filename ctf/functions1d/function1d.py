# Main Imports
import numpy as np
import matplotlib.pyplot as plt


#
class Function1D():
    """ One Dimensional problem class. """

    def __init__(self):
        self.min = np.array([0.0])
        self.value = 0.0
        self.domain = np.array([-10.0, 10.0])
        self.smooth = False
        self.info = [False, False, False]
        self.latex_name = "Undefined"
        self.latex_cost = "Undefined"
        self.latex_desc = "Undefined"
        self.cost = lambda x: 0
        self.grad = lambda x: np.array([0, 0])
        self.hess = lambda x: np.array([[0, 0], [0, 0]])

    def plot_cost(self, points=200):
        """ Plots the cost contour plot over the domain of the function. """
        ## Latex
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ## Domain Correction
        # Lower x0 Limit
        if np.isfinite(self.domain[0][0]):
            x_lim_lower = self.domain[0][0]
        else:
            x_lim_lower = -10.0
        # Upper x0 Limit
        if np.isfinite(self.domain[0][1]):
            x_lim_upper = self.domain[0][1]
        else:
            x_lim_upper = +10.0
        ## Lines
        x = np.linspace(x_lim_lower, x_lim_upper, points).reshape(1, -1)

        ## Calculate Costs
        cost = self.cost(x)
        ## Plot
        plt.figure()
        plt.plot(x.T, cost)
        plt.scatter(self.min, self.value, c='k', marker='x')
        plt.grid()
        plt.title(self.latex_name + "\n" + self.latex_cost)
        plt.subplots_adjust(top=0.8)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.xlim([x_lim_lower, x_lim_upper])


    def plot_grad(self, points=40):
        """ Plots the grad quiver plot over the domain of the function. """
        ## Latex
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ## Domain Correction
        # Lower x0 Limit
        if np.isfinite(self.domain[0][0]):
            x_lim_lower = self.domain[0][0]
        else:
            x_lim_lower = -10.0
        # Upper x0 Limit
        if np.isfinite(self.domain[0][1]):
            x_lim_upper = self.domain[0][1]
        else:
            x_lim_upper = +10.0
        ## Lines
        x = np.linspace(x_lim_lower, x_lim_upper, points).reshape(1, -1)
        ## Calculate Grad
        grad = self.grad(x)
        ## Plot
        plt.figure()
        plt.plot(x.T, grad.T, color='red')
        plt.scatter(self.min, self.value, c='k', marker='x')
        plt.grid()
        plt.title(self.latex_name + "\n" + self.latex_cost)
        plt.subplots_adjust(top=0.8)
        plt.xlabel('$x$')
        plt.ylabel(r'$\nabla f(x)$')
        plt.xlim([x_lim_lower, x_lim_upper])


    def plot_both(self, points=200):
        """ Plots the grad quiver plot over the domain of the function. """
        ## Latex
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ## Domain Correction
        # Lower x Limit
        if np.isfinite(self.domain[0][0]):
            x_lim_lower = self.domain[0][0]
        else:
            x_lim_lower = -10.0
        # Upper x Limit
        if np.isfinite(self.domain[0][1]):
            x_lim_upper = self.domain[0][1]
        else:
            x_lim_upper = +10.0
        ## Lines
        x = np.linspace(x_lim_lower, x_lim_upper, points).reshape(1, -1)

        ## Calculate Costs
        cost = self.cost(x)
        ## Calculate Grad
        grad = self.grad(x)
        ## Plot
        plt.figure()
        plt.plot(x.T, cost, color='blue')
        plt.plot(x.T, grad.T, color='red')
        plt.scatter(self.min, self.value, c='k', marker='x')
        plt.grid()
        plt.title(self.latex_name + "\n" + self.latex_cost)
        plt.subplots_adjust(top=0.8)
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
        plt.xlim([x_lim_lower, x_lim_upper])
