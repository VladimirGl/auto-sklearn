# -*- encoding: utf-8 -*-
from autosklearn.util import dependencies
from autosklearn.__version__ import __version__


__MANDATORY_PACKAGES__ = '''
numpy>=1.9
scikit-learn>=0.19,<0.20
lockfile>=0.10
smac==0.3.0
pyrfr==0.2.0
ConfigSpace>=0.3.1,<0.4
xgboost
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)
