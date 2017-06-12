# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:09:23 2017

@author: Harry
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class likelihoodModel:
    """Abstract class for likelihood models"""
    
    __metaclass__ = ABCMeta
    
    def __init__(self, y, X, weights=[]):
        """
        Initializer for likelihood model
        
        param y ndarray: response variable
        param X ndarray: vector of features
        param weights ndarray: vector of weights for the observations               
        """
        self.N, self.K = X.shape
        self.y = y
        self.X = np.concatenate((np.ones((self.N,1)),X),axis=1)
        if weights == []:
            self.weights = np.ones((self.N,1))
        else:
            self.weights = weights * self.N

    def lossFunction(self, beta):
        """returns the negative ll to minimize"""
        return -self.loglikelihood(beta)
        
    def lossGradient(self, beta):
        """returns the negative jacobian for minimization"""        
        return -self.jacobian(beta)
        
    def lossHessian(self, beta):
        """returns the negative hessian for minimization"""
        return -self.hessian(beta)

    @abstractmethod
    def loglikelihood(self,beta):
        """Likelihood function"""
        raise Exception("Likelihood should be implemented in child class")
    
    @abstractmethod    
    def jacobian(self,beta):
        """Method returns the first derivative of the likelihood"""
        raise Exception("Likelihood should be implemented in child class")
        
    @abstractmethod
    def hessian(self,beta):       
        raise Exception("Likelihood should be implemented in child class")