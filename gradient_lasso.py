# -*- coding: utf-8 -*-
"""
@author: Harry
"""
import numpy as np
from scipy import optimize
import copy

class gradientLasso(object):
    """Class implements gradient Lasso"""
    def __init__(self,model):
        """
        Initialize class
        
        param model likelihoodModel: model to estimate with lasso
        """
        self.model = model
        self.nParam = model.K
        
    def getLoss(self,beta):
        """returns loss"""
        return self.model.lossFunction(beta)

    def getGradient(self,beta):
        """returns loss"""
        return self.model.lossGradient(beta)

    def alphaLoss(self,alpha,w_ko,v,constraint):
        """Helper function finds optimal alpha at Addition step"""
        alpha_tr = max(min(alpha,1),0)
        loss = self.getLoss(constraint * ((1-alpha_tr) * w_ko + alpha_tr * v))
        if np.isnan(loss):
            return np.inf
        else:
            return loss
   
    def deltaLoss(self,delta,w_k,h,L,constraint):
        """Helper function finds optimal delta at Deletion step"""
        delta_tr = max(min(delta,L),0.0)
        loss = self.getLoss(constraint * (w_k + delta_tr * h))
        if np.isnan(loss):
            return np.inf
        else:
            return loss
    
    def estimate(self, constraint, prec_tol=1e-9, maxIter=1e4 ):
        """run the gradient lasso algorithm"""
        error, Ones = np.inf, np.ones(self.nParam)
        w_k = np.zeros(self.nParam)
        cpt = 0
        alpha, delta = 1.0 / constraint, 0.1
        llo = self.getLoss(constraint * w_k)
        while abs(error) > prec_tol and cpt<=maxIter:
            # 1. Addition Step:
            w_ko = copy.deepcopy(w_k)
            v = np.zeros(self.nParam)
            grad1 = constraint * self.getGradient(constraint * w_k)
            k_hat = np.argmax(np.abs(grad1))
            gamma = -1.0 * np.sign(grad1[k_hat])
            v[k_hat] = gamma
            alpha_s = optimize.fmin(self.alphaLoss,
                                    alpha,
                                    args=(w_ko,v,constraint),
                                    xtol=1e-9, 
                                    disp=False)
            alpha = max(min(alpha_s[0],1.0),0.0)
            w_k = (1-alpha) * w_ko + alpha * v
            
            #2. Deletion Step:
            h = np.zeros(self.nParam)
            grad2 = constraint * self.getGradient(constraint * w_k)
            theta = np.sign(w_k)
            sigma = (w_k!=0)
            grad_sig = grad2[sigma]
            theta_sig = theta[sigma]
            if grad_sig.dot(theta_sig)<0.0 and theta.dot(w_k)==1:
                h_sig = -grad_sig + theta_sig.dot(grad_sig)/Ones[sigma].sum() * theta_sig
            else:
                h_sig = -grad_sig
            h[sigma] = h_sig
            setL = [-ww/h[idx] for idx,ww in enumerate(w_k) if ww*h[idx]<0.0 and sigma[idx]]
            if h_sig.dot(theta_sig) >0.0:
                L = min(setL + [(1.0-np.abs(w_k).sum())/(h_sig.dot(theta_sig))])
            else:
                L = min(setL)
            delta_s = optimize.fmin(self.deltaLoss,
                                    delta,
                                    args=(w_k,h,L,constraint),
                                    xtol=1e-9,
                                    disp=False)
            delta = max(min(delta_s[0],L),0.0)
            w_k += delta*h
            error = self.getLoss(constraint * w_k) - llo
            llo = error + llo
            cpt+=1
        return w_k*constraint

