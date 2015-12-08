from collections import namedtuple
from functools import wraps

import numpy

import nose.tools as nt

from paramnormal import fit

@nt.nottest
def seed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)
    return wrapper


@nt.nottest
def assert_close_enough(actual, expected):
    nt.assert_almost_equal(actual, expected, places=5)


def test__pop_none():
    nt.assert_dict_equal(
        fit._pop_none(a=None, b=1, c=None),
        dict(b=1)
    )


class Test_uniform(object):
    @seed
    def setup(self):
        self.data = numpy.random.uniform(low=2.0, high=6.7, size=100)
        self.fitter = fit.uniform

    @nt.nottest
    @staticmethod
    def check_params(params, expected_low, expected_high):
        assert_close_enough(params.low, expected_low)
        assert_close_enough(params.high, expected_high)

    def test_min_guesses(self):
        self.check_params(
            self.fitter(self.data),
            1.6794396946326486,
            6.6453579038130126
        )

    def test_guess_low(self):
        self.check_params(
            self.fitter(self.data, low=2.0),
            2.0220602950494859,
            6.6453579038130126
        )

    def test_guessboth(self):
        self.check_params(
            self.fitter(self.data, low=2.0, high=6.7),
            2.0220657091381238,
            6.6453579038130126
        )

    def test_guess_width(self):
        self.check_params(
            self.fitter(self.data, width=4.7),
            2.0220602950494859,
            6.6453579038130126
        )


class  Test_normal(object):
    @seed
    def setup(self):
        self.data = numpy.random.normal(loc=2.0, scale=6.7, size=100)
        self.fitter = fit.normal

    @nt.nottest
    @staticmethod
    def check_params(params, expected_mu, expected_sigma):
        assert_close_enough(params.mu, expected_mu)
        assert_close_enough(params.sigma, expected_sigma)

    def test_min_guesses(self):
        self.check_params(
            self.fitter(self.data),
            2.4007137040810496,
            6.7528110396010854
        )