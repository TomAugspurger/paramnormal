from collections import namedtuple
from functools import wraps

import numpy

import nose.tools as nt

from paramnormal import fit


def seed(func):
    @wraps(func):
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)
    return wrapper


def test__pop_none():
    nt.assert_dict_equal(
        fit._pop_none(a=None, b=1, c=None),
        dict(b=1)
    )


class Test_uniform():
    @seed
    def setup(self):
        self.data = numpy.random.uniform(low=2.0, high=6.7, size=100)
        self.fitter = fit.uniform

    def test_min_guesses(self):
        params = self.fitter(self.data)
        nt.assert_almost_equal(params.low, 2.0108025429305361)
        nt.assert_almost_equal