from collections import namedtuple

import numpy
from scipy import stats

from . import process_args


_docstring = """\

    Parameters
    ----------
    data : array-like
        Data to be fit to the distribution.
    guesses : keyword arguments, optional
        Initial guesses for the distribution parameters.

    Returns
    -------
    params : namedtuple
        Collection of parameters that define the distribution in a
        manner consistent with ``paramnormal``.

"""


def _pop_none(**kwargs):
    final = kwargs.copy()
    for k in kwargs:
        if kwargs[k] is None:
            final.pop(k)
    return final


def _fit(scipyname, data, pnormname=None, **guesses):
    if pnormname is None:
        pnormname = scipyname

    dist = getattr(stats, scipyname)
    processor = getattr(process_args, pnormname)
    args = _pop_none(**processor(**guesses))
    return dist.fit(data, **args)


def uniform(data, **guesses):
    """ Fit a uniform distribution to data.
    {}""".format(_docstring)

    params = _fit('uniform', data, **guesses)
    template = namedtuple('params', ['low', 'high'])
    return template(low=params[0], high=params[1] + params[0])


def normal(data, **guesses):
    """ Fit a normal distribution to data.
    {}""".format(_docstring)

    params = _fit('norm', data, pnormname='normal', **guesses)
    template = namedtuple('params', ['mu', 'sigma'])
    return template(*params)


def lognormal(data, **guesses):
    """ Fit a lognormal distribution to data.
    {}""".format(_docstring)

    params =  _fit('lognorm', data, pnormname='lognormal', **guesses)
    template = namedtuple('params', ['mu', 'sigma', 'offset'])
    return template(mu=numpy.log(params[2]), sigma=params[0], offset=params[1])


def beta(data, **guesses):
    """ Fit a beta distribution to data.
    {}""".format(_docstring)

    params = _fit('beta', data, **guesses)
    template = namedtuple('params', ['alpha', 'beta'])
    return template(*params)


def chi_squared(data, **guesses):
    """ Fit a chi_squared distribution to data.
    {}""".format(_docstring)

    params = _fit('chi2', data, pnormname='chi_squared',**guesses)
    template = namedtuple('params', ['k'])
    return template(*params)


def pareto(data, **guesses):
    """ Fit a pareto distribution to data.
    {}""".format(_docstring)

    params = _fit('pareto', data, **guesses)
    template = namedtuple('params', ['alpha'])
    return template(*params)


def gamma(data, **guesses):
    """ Fit a gamma distribution to data.
    {}""".format(_docstring)

    params = _fit('gamma', data, **guesses)
    template = namedtuple('params', ['k', 'theta'])
    return template(*params)

