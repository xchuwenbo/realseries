# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:07:05 2020

@author: zhihengzhang
"""
import sys,os

import realseries
import matplotlib as plt
import numpy as np
realseries.__file__

from realseries.models.base import BaseModel
from realseries.models.NAR import NAR_Network
from realseries.models.AR import AR
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR


class GC(BaseModel):
    
    '''
        Args:
            win_len: window length
            model: 'AR' or 'NAR-network'
            method: option of fitting method, 'NAR'/'AR'
        Attributes:
             -- single_error: the fitting error without other channel's dimension
             -- double_error: the fitting error with other channel's dimension
    '''
    
    global input_list
    global target_list
    def __init__(self, win_len , model, method,train_rate):
        self.win_len = win_len

        self.model = model
        self.method = method       
        
        self.train_rate = train_rate
        pass
    def fit(self,X,y=None):
        pass
        

    def detect(self,X):
        """

        Args:
            X : time series pair

        Returns:
            Ftest_win : window-level causality

        """
        
        X0 = X[0]
        X1 = X[1]
        if self.method =='NAR':
        

            GC_network1 = NAR_Network(self.model.inodes,self.model.hnodes,self.model.onodes,self.model.lr)
            GC_network1.fit((X[0])[0:int(self.train_rate * len(X[0]))]) 
            single_error = GC_network1.detect(X[0])   #inputnodes ~ len(X0)
        
            GC_network2 = NAR_Network(self.model.inodes*2,self.model.hnodes,self.model.onodes,self.model.lr)
            GC_network2.fit( [ (X[0])[0:int(self.train_rate * len(X[0]))], (X[1])[0:int(self.train_rate * len(X[1]))]  ])    
            double_error = GC_network2.detect(X)
        

        
            single_error  = single_error[0]
            double_error = double_error[0]
            
            win_number = int((len(X[0])-self.model.inodes)/self.win_len)
        
        if self.method == 'AR':
            

            GC_network1 = AR(self.model.lag)     
            
            single_error = (np.abs(GC_network1.detect(X0)-X0))[0]
            
            GC_network2 = AR(self.model.lag*2)    
            
            double_error =  (np.abs(GC_network1.detect(X)-X0))[0]
            

            win_number = int((len(X[0])-self.model.lag)/self.win_len)
        

        Ftest_win =[]
        for i in range(win_number): 
            error1_win = 0
            error2_win = 0
            for j in range(self.win_len):
                error1_win += np.abs(single_error[i * self.win_len +j])
                error2_win += np.abs(double_error[i * self.win_len +j])

            Ftest_win.append( error1_win /error2_win)
        

        
            
        
        return Ftest_win
        
