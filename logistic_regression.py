# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:09:53 2017

Simple class for weighted logistic regression

@author: Harry
"""
from likelihood_model import likelihoodModel
import numpy as np
from scipy import optimize

class logisticRegression(likelihoodModel):
    """class for logistic reg."""
    
    def __init__(self, y, X, weights=[]):
        """
        Initializer for logistic regression
        
        param y ndarray: response variable
        param X ndarray: vector of features
        param weights ndarray: vector of weights for the observations               
        """
        self.N = X.shape[0]
        self.y = y
        self.X = np.concatenate((np.ones((self.N,1)),X),axis=1)
        if weights == []:
            self.weights = np.ones((self.N,1))
        else:
            self.weights = weights * self.N
        self.K = self.X.shape[1]
           
    
    def logstic(self,beta):
        """Logistic transform"""
        return 1.0/(1.0 + np.exp(-self.X.dot(beta))).reshape(self.N,1)
        
    def loglikelihood(self,beta):
        """Likelihood function"""
        y_hat = self.logstic(beta)
        return (self.weights*(self.y * np.log(y_hat) + (1-self.y)*np.log(1-y_hat))).sum()

    def jacobian(self,beta):
        """Method returns the first derivative of the likelihood"""
        y_hat = self.logstic(beta)
        return (self.weights*(self.y - y_hat) * self.X).sum(axis=0)

    def hessian(self,beta):
        """Method returns the second derivative of the likelihood"""
        y_hat = self.logstic(beta)
        return -(self.weights * self.X * y_hat * (1-y_hat)).T.dot(self.X)
        
    def estimate(self):
        """Method runs ML estimation"""
        x0 = np.random.randn(self.K,1)
        beta = optimize.fmin_ncg(self.lossFunction,
                                 x0,
                                 fprime = self.lossGradient,
                                 fhess = self.lossHessian,
                                 avextol = 1e-9,
                                 )
        return beta