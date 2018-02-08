# -*- encoding: utf-8 -*-
import setuptools
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize

extensions = cythonize(
    [Extension('autosklearn.data.competition_c_functions',
               sources=['autosklearn/data/competition_c_functions.pyx'],
               language='c',
               include_dirs=[np.get_include()])
     ])

requirements = [
    "setuptools",
    "nose==1.3.7",
    "six==1.11.0",
    "Cython==0.27.3",
    "numpy>=1.9.0",
    "scipy>=0.14.1",
    "scikit-learn>=0.19,<0.20",
    "lockfile==0.12.2",
    "joblib==0.11",
    "psutil==5.4.3",
    "pyyaml==3.12",
    "liac-arff==2.1.1",
    "pandas==0.19.2",
    "ConfigSpace>=0.4.0,<0.5",
    "pynisher>=0.4,<0.5",
    "pyrfr>=0.6.1,<0.8",
    "smac>=0.8,<0.9",
    "xgboost==0.7.post3"
]

with open("autosklearn/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setuptools.setup(
    name='auto-sklearn',
    description='Automated machine learning.',
    version=version,
    ext_modules=extensions,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    test_suite='nose.collector',
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    url='https://automl.github.io/auto-sklearn')
