# Imports
import numpy as np
from ctf.functions2d import *


# Support Functions
def check_cost(f):
    """ Checks cost is within tolerances. """
    return (f.cost(f.min) - f.value) < 1e-4


def check_grad(f):
    """ Checks cost is within tolerances. """
    return np.linalg.norm(f.grad(f.min)) < 1e-4


def check_hess(f):
    """ Checks Hessian is positive semi-definite. """
    #try:
    #    np.linalg.cholesky(f.hess(f.min))
    #except:
    #    return False
    #else:
    #    return True
    return np.all(np.linalg.eigvals(f.hess(f.min)) >= 0.0)
    #return True


# Test Functions
# class TestAckley():
#
#     def setup(self):
#         self.f = Ackley()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)


class TestBeale():

    def setup(self):
        self.f = Beale()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


class TestBooth():

    def setup(self):
        self.f = Booth()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


# class TestBukin():
#
#     def setup(self):
#         self.f = Bukin()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)


# class TestCrossInTray():
#
#     def setup(self):
#         self.f = CrossInTray()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)


class TestEasom():

    def setup(self):
        self.f = Easom()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


# class TestEggholder():
#
#     def setup(self):
#         self.f = Eggholder()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)


class TestGoldsteinPrice():

    def setup(self):
        self.f = GoldsteinPrice()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


# class TestHolderTable():
#
#     def setup(self):
#         self.f = HolderTable()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)


class TestLevy13():

    def setup(self):
        self.f = Levy13()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


class TestMatyas():

    def setup(self):
        self.f = Matyas()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


class TestMcCormick():

    def setup(self):
        self.f = McCormick()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


class TestRosenbrock():

    def setup(self):
        self.f = Rosenbrock()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


# class TestSchaffer2():
#
#     def setup(self):
#         self.f = Schaffer2()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)


class TestSphere():

    def setup(self):
        self.f = Sphere()

    def teardown(self):
        pass

    def test_cost(self):
        """ Test if the cost at the minimum is what is expected. """
        assert check_cost(self.f)

    def test_grad(self):
        """ Test if the gradient at the minimum is sufficiently close to zero. """
        assert check_grad(self.f)

    def test_hess(self):
        """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
        assert check_hess(self.f)


# class TestStyblinskiTang():
#
#     def setup(self):
#         self.f = StyblinskiTang()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)


# class TestThreeHumpCamel():
#
#     def setup(self):
#         self.f = ThreeHumpCamel()
#
#     def teardown(self):
#         pass
#
#     def test_cost(self):
#         """ Test if the cost at the minimum is what is expected. """
#         assert check_cost(self.f)
#
#     def test_grad(self):
#         """ Test if the gradient at the minimum is sufficiently close to zero. """
#         assert check_grad(self.f)
#
#     def test_hess(self):
#         """ Test if the gradient at the minimum is sufficiently positive semi-definite. """
#         assert check_hess(self.f)

