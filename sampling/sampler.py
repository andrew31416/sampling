"""
sampler.py
============================================================================
Routines to perform sample-diversification for mini-batch stochastic gradient
descent/ascent.
"""
from sklearn.gaussian_process.kernels import RBF
import numpy as np


class sampler():
    """
    Base class for mini-batch sampler of categorical data.
    """

    def set_X(self,X):
        """
        The input descriptor X can be a 1-d or 1-d array-like structure. If the
        array is 1-d, the array is reshaped to (-1,1).

        Arguments
        ---------
        X : (list,np.ndarray)
            An array-like object of the complete data descriptors.
        """
        if not isinstance(X,(list,np.ndarray)) or len(np.shape(X))>2:
            raise Exception(self.set_X.__doc__)
        
        self.X = X
        if isinstance(X,list):
            self.X = np.asarray(self.X)

    def set_y(self,y):
        """
        The categorical labels for the complete data set. Elements of y must
        be integer.

        Arguments
        ---------
        y : (list,np.ndarray)
            Integer categorical labels.
        """
        if not isinstance(y,(list,np.ndarray)) or len(np.shape(y))>1:
            raise Exception(self.set_y.__doc__)
        
        self.y = y
        if isinstance(y,list):
            self.y = np.asarray(y)

