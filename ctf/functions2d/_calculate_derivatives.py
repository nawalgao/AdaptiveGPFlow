from sympy import *

# Set-Up
x, y = symbols('x[0], x[1]')


# Problems
ackley = -20*exp(-0.2*sqrt(0.5*(x**2 + y**2))) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + 20 + exp(1)
beale = (1.5 - x + x*y)**2.0 + (2.25 - x + x*y**2.0)**2.0 + (2.625 - x + x*y**3.0)**2.0
booth = (x + 2*y - 7)**2 + (2*x + y - 5)**2
bukin = 100*sqrt(abs(y - 0.01*x**2)) + 0.01*abs(y + 10)
cross_in_tray = 0
easom = -cos(x)*cos(y)*exp(-((x-pi)**2 + (y-pi)**2))
eggholder = -(y + 47)*sin(sqrt(abs(y + x/2 + 47))) - x*sin(sqrt(abs(x-(y+47))))
goldstein_price = (1+((x+y+1)**2)*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+((2*x-3*y)**2)*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
levi_13 = sin(3*pi*x)**2+((x-1)**2)*(1+sin(3*pi*y)**2)+(1+sin(2*pi*y)**2)*(y-1)**2
matyas = 0.26*(x**2 + y**2) - 0.48*x*y
mccormick = sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1


# Functions
def derivatives(problem):
    dx = diff(problem, x)
    dy = diff(problem, y)
    dxx = diff(dx, x)
    dxy = diff(dx, y)
    dyy = diff(dy, y)
    # Print Results
    print('f:')
    print(problem)
    print('dx:')
    print(dx)
    print('dy:')
    print(dy)
    print('dxx:')
    print(dxx)
    print('dxy:')
    print(dxy)
    print('dyy:')
    print(dyy)


# Test
derivatives(ackley)

