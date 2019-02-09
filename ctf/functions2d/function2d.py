# Main Imports
import numpy as np
import matplotlib.pyplot as plt


#
class Function2D():
    """ Two Dimensional problem class. """

    def __init__(self):
        self.min = np.array([0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-10.0, 10.0], [-10.0, 10.0]])
        self.smooth = False
        self.info = [False, False, False]
        self.latex_name = "Undefined"
        self.latex_type = "Undefined"
        self.latex_cost = "Undefined"
        self.latex_desc = "Undefined"
        self.cost = lambda x: 0
        self.grad = lambda x: np.array([0, 0])
        self.hess = lambda x: np.array([[0, 0], [0, 0]])

    def plot_cost(self, points=200):
        """ Plots the cost contour plot over the domain of the function. """
        # Latex
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ## Domain Correction
        # Lower x0 Limit
        if np.isfinite(self.domain[0][0]):
            x0_lim_lower = self.domain[0][0]
        else:
            x0_lim_lower = -10.0
        # Upper x0 Limit
        if np.isfinite(self.domain[0][1]):
            x0_lim_upper = self.domain[0][1]
        else:
            x0_lim_upper = +10.0
        # Lower x1 Limit
        if np.isfinite(self.domain[1][0]):
            x1_lim_lower = self.domain[1][0]
        else:
            x1_lim_lower = -10.0
        # Upper x1 Limit
        if np.isfinite(self.domain[1][1]):
            x1_lim_upper = self.domain[1][1]
        else:
            x1_lim_upper = +10.0
        ## Lines
        x0 = np.linspace(x0_lim_lower, x0_lim_upper, points)
        x1 = np.linspace(x1_lim_lower, x1_lim_upper, points)
        ## Meshes
        X0, X1 = np.meshgrid(x0, x1)
        ## Combined
        X = np.array([X0, X1])
        ## Calculate Costs
        cost = self.cost(X)
        ## Renormalise
        cost_norm = np.log(cost - np.min(cost) + 1)
        ## Plot
        plt.figure()
        plt.contourf(X0, X1, cost_norm, 50)
        plt.scatter(self.min[..., 0], self.min[..., 1], c='w', marker='x')
        plt.grid()
        plt.title(self.latex_name + "\n" + self.latex_cost)
        plt.subplots_adjust(top=0.8)
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
        plt.xlim([x0_lim_lower, x0_lim_upper])
        plt.ylim([x1_lim_lower, x1_lim_upper])

    def plot_grad(self, points=200):
        """ Plots the grad quiver plot over the domain of the function. """
        # Latex
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ## Domain Correction
        # Lower x0 Limit
        if np.isfinite(self.domain[0][0]):
            x0_lim_lower = self.domain[0][0]
        else:
            x0_lim_lower = -10.0
        # Upper x0 Limit
        if np.isfinite(self.domain[0][1]):
            x0_lim_upper = self.domain[0][1]
        else:
            x0_lim_upper = +10.0
        # Lower x1 Limit
        if np.isfinite(self.domain[1][0]):
            x1_lim_lower = self.domain[1][0]
        else:
            x1_lim_lower = -10.0
        # Upper x1 Limit
        if np.isfinite(self.domain[1][1]):
            x1_lim_upper = self.domain[1][1]
        else:
            x1_lim_upper = +10.0
        ## Lines
        x0 = np.linspace(x0_lim_lower, x0_lim_upper, points)
        x1 = np.linspace(x1_lim_lower, x1_lim_upper, points)
        ## Meshes
        X0, X1 = np.meshgrid(x0, x1)
        ## Combined
        X = np.array([X0, X1])
        ## Calculate Grad
        grad = self.grad(X)
        ## Renormalise
        grad_norm = grad / np.log(1+np.linalg.norm(grad, axis=0))
        grad_norm = grad / np.linalg.norm(grad, axis=0)
        ## Plot
        plt.figure()
        plt.quiver(X0, X1, -grad_norm[0], -grad_norm[1])
        plt.scatter(self.min[..., 0], self.min[..., 1], c='w', marker='x')
        plt.grid()
        plt.title(self.latex_name + "\n" + self.latex_cost)
        plt.subplots_adjust(top=0.8)
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
        plt.xlim([x0_lim_lower, x0_lim_upper])
        plt.ylim([x1_lim_lower, x1_lim_upper])


    def plot_both(self, c_points=200, g_points=200):
        """ Plots the grad quiver plot over the domain of the function. """
        # Latex
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ## Domain Correction
        # Lower x0 Limit
        if np.isfinite(self.domain[0][0]):
            x0_lim_lower = self.domain[0][0]
        else:
            x0_lim_lower = -10.0
        # Upper x0 Limit
        if np.isfinite(self.domain[0][1]):
            x0_lim_upper = self.domain[0][1]
        else:
            x0_lim_upper = +10.0
        # Lower x1 Limit
        if np.isfinite(self.domain[1][0]):
            x1_lim_lower = self.domain[1][0]
        else:
            x1_lim_lower = -10.0
        # Upper x1 Limit
        if np.isfinite(self.domain[1][1]):
            x1_lim_upper = self.domain[1][1]
        else:
            x1_lim_upper = +10.0
        ## Lines
        x0c = np.linspace(x0_lim_lower, x0_lim_upper, c_points)
        x1c = np.linspace(x1_lim_lower, x1_lim_upper, c_points)
        x0g = np.linspace(x0_lim_lower, x0_lim_upper, g_points)
        x1g = np.linspace(x1_lim_lower, x1_lim_upper, g_points)
        ## Meshes
        X0c, X1c = np.meshgrid(x0c, x1c)
        X0g, X1g = np.meshgrid(x0g, x1g)
        ## Combined
        Xc = np.array([X0c, X1c])
        Xg = np.array([X0g, X1g])
        ## Calculate Costs
        cost = self.cost(Xc)
        ## Renormalise
        cost_norm = np.log(cost - np.min(cost) + 1)
        ## Calculate Grad
        grad = self.grad(Xg)
        ## Renormalise
        grad_norm = grad / np.linalg.norm(grad, axis=0)
        ## Plot
        plt.figure()
        plt.contourf(X0c, X1c, cost_norm, 50)
        plt.scatter(self.min[..., 0], self.min[..., 1], c='w', marker='x')
        plt.streamplot(X0g, X1g, -grad_norm[0], -grad_norm[1], density=4.0, color='k')
        plt.scatter(self.min[0], self.min[1], c='w', marker='x')
        plt.grid()
        plt.title(self.latex_name + "\n" + self.latex_cost)
        plt.subplots_adjust(top=0.8)
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
        plt.xlim([x0_lim_lower, x0_lim_upper])
        plt.ylim([x1_lim_lower, x1_lim_upper])
