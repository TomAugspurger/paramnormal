# Setup script for the paramnormal package
#
# Usage: python setup.py install

import os
from setuptools import setup, find_packages


DESCRIPTION = "paramnormal: Conventionally parameterized probability distributions"
LONG_DESCRIPTION = DESCRIPTION
NAME = "paramnormal"
VERSION = "0.1"
AUTHOR = "Paul Hobson"
AUTHOR_EMAIL = "phobson@geosyntec.com"
URL = "https://github.com/phobson/paramnormal"
DOWNLOAD_URL = "https://github.com/phobson/paramnormal/archive/master.zip"
LICENSE = "BSD 3-clause"
PACKAGES = find_packages()
PLATFORMS = "Python 3.3 and later."
CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Intended Audience :: Science/Research",
    "Topic :: Formats and Protocols :: Data Formats",
    "Topic :: Scientific/Engineering :: Earth Sciences",
    "Topic :: Software Development :: Libraries :: Python Modules",
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
]
INSTALL_REQUIRES = ['numpy', 'scipy']
PACKAGE_DATA = {}
DATA_FILES = []


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        platforms=PLATFORMS,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
