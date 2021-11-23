# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:34:04 2020

@author: zhihengzhang
"""

import sys,os

import realseries
import matplotlib as plt
import numpy as np
realseries.__file__


from realseries.models.base import BaseModel


from matplotlib import pyplot



import pandas as pd
from pandas import DataFrame,Series

from sklearn.linear_model import LinearRegression
 

class AR(BaseModel):
    """
    Args:
        lag: the time lag in AR model
        
    Attributes: 
        X_train: training data in AR model.
        Y_train: training label in AR model.

    """
    def __init__(self,lag):
        self.lag = lag

        pass
    
    def fit(self,X,y = None):
        """
        Args:
            X : time series(dimension is 1 or 2)
            y :  The default is None.

        Returns:
            None.

        """
        X_array = np.array(X)

        if X_array.shape[0] !=2:  
            max_X0 = max(X);min_X0 = min(X)

            for i in range(0,len(X)):
                X[i] = (X[i]-min_X0)/(max_X0-min_X0)
            
            A=[]; lag = self.lag  
            for i in range(0, len(X)-lag):
                temp = []
                for j in range(0, lag):
                    temp.append(X[i+j])
                A.append(temp)
            input_list = A
        
            B=[];
            for i in range(0, len(X)-lag):
                temp = []
                temp.append(X[i+lag])
                B.append(temp)
            target_list = B

        
        if X_array.shape[0] == 2:
            X0 = X[0]
            X1 = X[1]

            max_X0 = max(X0);min_X0 = min(X0)
            
            for i in range(0,len(X0)):
                X0[i] = (X0[i]-min_X0)/(max_X0-min_X0)
                
                
                
            max_X1 = max(X1);min_X1 = min(X1)
          
            for i in range(0,len(X1)):
                X1[i] = (X1[i]-min_X1)/(max_X1-min_X1)
 
                
            lag=self.lag    
            
            A=[]; lag=int(lag/2)    
            

            for i in range(0, len(X0)-lag):
                temp = []
                for j in range(0, lag):
                    temp.append(X0[i+j])
                for j in range(0, lag):
                    temp.append(X1[i+j])   
                A.append(temp)
            input_list = A   
            
            B=[];
            for i in range(0, len(X0)-lag):
                temp = []
                temp.append(X0[i+lag])
                B.append(temp)
            target_list = B
        
            
        X_train = np.matrix(input_list)
         
        Y_train = np.matrix(target_list)
         


        
        model = LinearRegression()
        
        model.fit(X_train,Y_train)
        
        a  = model.intercept_
        b = model.coef_

 
        
        pass
    
    def detect(self,X):
        """
        Args:
            X : time series(dimension is 1 or 2)
            y :  The default is None.

        Returns:
            detection: the fitting result of training data.
            
        """
        X_array = np.array(X)

        if X_array.shape[0] !=2:  
            max_X0 = max(X);min_X0 = min(X)

            for i in range(0,len(X)):
                X[i] = (X[i]-min_X0)/(max_X0-min_X0)
            
            A=[]; lag = self.lag  
            for i in range(0, len(X)-lag):
                temp = []
                for j in range(0, lag):
                    temp.append(X[i+j])
                A.append(temp)
            input_list = A
        
            B=[];
            for i in range(0, len(X)-lag):
                temp = []
                temp.append(X[i+lag])
                B.append(temp)
            target_list = B

        
        if X_array.shape[0] == 2:
            X0 = X[0]
            X1 = X[1]

            max_X0 = max(X0);min_X0 = min(X0)
            
            for i in range(0,len(X0)):
                X0[i] = (X0[i]-min_X0)/(max_X0-min_X0)
                
                
                
            max_X1 = max(X1);min_X1 = min(X1)
          
            for i in range(0,len(X1)):
                X1[i] = (X1[i]-min_X1)/(max_X1-min_X1)
               
                
            lag=self.lag    
            
            A=[]; lag=int(lag/2)    
            
            for i in range(0, len(X0)-lag):
                temp = []
                for j in range(0, lag):
                    temp.append(X0[i+j])
                for j in range(0, lag):
                    temp.append(X1[i+j])   
                A.append(temp)
            input_list = A   
            
            B=[];
            for i in range(0, len(X0)-lag):
                temp = []
                temp.append(X0[i+lag])
                B.append(temp)
            target_list = B
        
            
        X_train = np.matrix(input_list)
         
        Y_train = np.matrix(target_list)
         

        
        model = LinearRegression()   
        
        model.fit(X_train,Y_train)
        
        a  = model.intercept_
        b = model.coef_

 
        
        detection = model.predict(X_train)
        
        detection = detection *(max_X0-min_X0) + min_X0
        
        
        return detection
    













