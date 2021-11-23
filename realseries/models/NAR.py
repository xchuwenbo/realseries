# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:36:45 2020

@author: studyzzh
"""

import sys,os

import pandas as pd
import realseries
import matplotlib.pyplot as plt
import numpy as np
realseries.__file__


from realseries.models.base import BaseModel




import numpy
import scipy.special


class NAR_Network(BaseModel):
    '''
        Args:
            -- inputnodes: the number of nodes in input layer.
            -- hiddennodes: the number of nodes in hidden layer.
            -- outputnodes: the number of nodes in outout layer.
            -- learning rate: the learning rate of NAR model.
        Attributes:
            -- fit_X : the fitting results on training data.
    '''

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate
        
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
       
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
   
    global input_list
    global target_list

    def fit(self,X,y=None):
        """
        Args:
            X : the input time series in shape of (,1) or (,2)
            y : The default is None.

        """
        
        global input_list
        global target_list 
        global max_X0
        global min_X0
        
        X_array = np.array(X)

        if X_array.shape[0] !=2:  
            max_X0 = max(X);min_X0 = min(X)

            for i in range(0,len(X)):
                X[i] = (X[i]-min_X0)/(max_X0-min_X0)
            
            A=[]; lag=self.inodes    
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

                
            lag=self.inodes   
            
            A=[]; lag=int(self.inodes/2)    
            

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
            

        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list,ndmin= 2).T

        fit_X = X;

        if X_array.shape[0] == 2:
            output_errors = [2]*(X_array.shape[1]-lag)
        else:
            output_errors = [2]*(len(X)-lag)
        epoch = 0
        while np.mean(np.abs(output_errors)) > 0.01 and epoch <10000:

            epoch += 1

            hidden_inputs = numpy.dot(self.wih, inputs)
            

            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = numpy.dot(self.who, hidden_outputs)
            final_outputs = self.activation_function(final_inputs)

            output_errors = targets - final_outputs
            hidden_errors = numpy.dot(self.who.T,output_errors)


            self.who += self.lr * numpy.dot((output_errors*final_outputs*(1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

            self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs*(1 - hidden_outputs)),
                                        numpy.transpose(inputs))


            
        fit_X = final_outputs*(max_X0-min_X0)+min_X0
        
        

  
        pass

    
    
    
    def detect(self, X):
        """
        Args:
            X : the input time series in shape of (,1) or (,2)
            y : The default is None.

        Returns:
            output_errors : the fitting errors on training data

        """

        
        global input_list
        global target_list 
        global max_X0
        global min_X0
        
        X_array = np.array(X)

        if X_array.shape[0] !=2:  
            max_X0 = max(X);min_X0 = min(X)

            for i in range(0,len(X)):
                X[i] = (X[i]-min_X0)/(max_X0-min_X0)
            
            A=[]; lag=self.inodes    
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

                
            lag=self.inodes   
            
            A=[]; lag=int(self.inodes/2)    
            
  
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
            

        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list,ndmin= 2).T
 
        fit_X = X;

        if X_array.shape[0] == 2:
            output_errors = [2]*(X_array.shape[1]-lag)
        else:
            output_errors = [2]*(len(X)-lag)
        epoch = 0
        while epoch ==0:

            epoch += 1

            hidden_inputs = numpy.dot(self.wih, inputs)
            

            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = numpy.dot(self.who, hidden_outputs)
            final_outputs = self.activation_function(final_inputs)

            output_errors = targets - final_outputs
            hidden_errors = numpy.dot(self.who.T,output_errors)


            self.who += self.lr * numpy.dot((output_errors*final_outputs*(1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

            self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs*(1 - hidden_outputs)),
                                        numpy.transpose(inputs))

             
        fit_X = final_outputs*(max_X0-min_X0)+min_X0
        fit_X = fit_X[0]
        if X_array.shape[0] != 2:
            fit_X_final = [0] * self.inodes
        if X_array.shape[0] == 2:
            fit_X_final = [0] * int(self.inodes/2)
        for i in range(len(fit_X)):
            fit_X_final.append(fit_X[i])        
 
        
        return output_errors

        
