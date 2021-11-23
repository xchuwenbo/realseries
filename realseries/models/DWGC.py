# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:57:41 2020

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


class DWGC(BaseModel):

    """
     Dynamic-window-level Granger Causality method, try to find the window-level causality on each channel pair.
     
         Args:
             win_len: window length
             model:AR or NAR
             index_lr: leanring rate of causal-indexing coefficients
             method: option of fitting method, 'NAR'/'AR'
         Attributes:
             causal_index : causal-indexing coefficients
             single_error1/single_error2 : the fitting error without other channel's dimension
             double_error: the fitting error with other channel's dimension
    """
    global input_list
    global target_list
    def __init__(self, win_len, model, index_lr,method,count,train_rate):
        self.win_len = win_len
        
        self.model = model
        
        self.index_lr = index_lr
        self.method = method
        self.count = count
        self.train_rate = train_rate
        pass
    def fit(self,X,y=None):
        pass
        

    def detect(self,X):
        """
        Args:
            X : the pair of time series

        Returns:
            Ftest_win : the causality on window-level
            
        """
        global Ftest_win
        global count 
        count = 0
        if self.method == 'NAR':
            win_number = int((len(X[0])-self.model.inodes)/self.win_len)  
            causality_index1 = [1]* len(X[0])  #initialization
            causality_index2 = [1]* len(X[0])
            double_error = [2] * (len(X[0])-self.model.inodes)
        

        
        
        
       
            while np.mean(np.abs(np.array((causality_index1)[self.model.inodes:win_number*self.win_len+self.model.inodes])-(np.array([np.mean(np.tanh(double_error))]*(win_number * self.win_len)-np.tanh(double_error[0:win_number*self.win_len])))) )> 0.1 and count<self.count:
            
            

            
                GC_network1 = NAR_Network(self.model.inodes,self.model.hnodes,self.model.onodes,self.model.lr)
                X[0] = np.array(causality_index1) * np.array(X[0])  
                X[0] = X[0].tolist();
              
                GC_network1.fit((X[0])[0:int(self.train_rate * len(X[0]))])
                single_error = GC_network1.detect(X[0])   
        
    
        
                GC_network2 = NAR_Network(self.model.inodes*2,self.model.hnodes,self.model.onodes,self.model.lr)
                X[1] = np.array(causality_index2) * np.array(X[1])
                X[1] = X[1].tolist()
                GC_network2.fit(  [ (X[0])[0:int(self.train_rate * len(X[0]))], (X[1])[0:int(self.train_rate * len(X[1]))]  ]   )
                double_error = GC_network2.detect(X)    

                
                GC_network1.fit((X[1])[0:int(self.train_rate * len(X[1]))])
                single_error_2 = GC_network1.detect(X[1])  
                
                single_error  = single_error[0]
                double_error = double_error[0]
                single_error_2  = single_error_2[0]
            
            

                Ftest_win =[]
                for i in range(win_number):  
                    error1_win = 0
                    error2_win = 0

                    for j in range(self.win_len):
                        error1_win += np.abs(single_error[i * self.win_len +j])*np.abs(single_error[i * self.win_len +j])
                        error2_win += np.abs(double_error[i * self.win_len +j])*np.abs(double_error[i * self.win_len +j])

                    Ftest_win.append(error1_win /error2_win)
        
         
          
        
            #optimize the causal-indexing coefficients
                for i in range(self.model.inodes, len(X[0])):
             
            
                    if int((i-self.model.inodes)/self.win_len)!=win_number:
                        '''
                        if Ftest_win[int((i-self.NAR_inputnodes)/self.win_len)] < 0.2:   #increase the recall:
                            if (np.mean(np.tanh(double_error))-np.tanh(double_error[i-self.NAR_inputnodes])) < 0:
                                (causality_index1)[i] = (causality_index1)[i]  + (1-self.index_lr)*(np.mean(np.tanh(double_error))-np.tanh(double_error[i-self.NAR_inputnodes]))
                            #print("causal_index[0]:",(causality_index1))
                            #print("causal_index[0][i-1]:",(causality_index[0])[i-1])
                                (causality_index2)[i] = (causality_index2)[i]  + (1-self.index_lr)*(np.mean(np.tanh(double_error))-np.tanh(double_error[i-self.NAR_inputnodes]))
                        '''
                    
                        if Ftest_win[int((i-self.model.inodes)/self.win_len)] > 2:   #increase the accuracy:
                            if (np.mean(np.tanh(single_error))-np.tanh(single_error[i-self.model.inodes])) < 0:
                                (causality_index1)[i] = (causality_index1)[i]  + (1-self.index_lr)*(np.mean(np.tanh(single_error))-np.tanh(single_error[i-self.model.inodes]))

                                (causality_index2)[i] = (causality_index2)[i]  + (1-self.index_lr)*(np.mean(np.tanh(single_error))-np.tanh(single_error[i-self.model.inodes]))    
                        
                    
                count+= 1

            
            
            
            
  
            
            
            
            
            
            
            
        if self.method == 'AR':
            win_number = int((len(X[0])-self.model.lag)/self.win_len)  
            causality_index1 = [1]* len(X[0])  #initialization
            causality_index2 = [1]* len(X[0])
            double_error = [2] * (len(X[0])-self.model.lag)
        

        
        
        
       
            while np.mean(np.abs(np.array((causality_index1)[self.model.lag:win_number*self.win_len+self.model.lag])-(np.array([np.mean(np.tanh(double_error))]*(win_number * self.win_len)-np.tanh(double_error[0:win_number*self.win_len])))) )> 0.1 and count<self.count:
            


            
                GC_network1 = AR(self.model.lag)
                X[0] = np.array(causality_index1) * np.array(X[0])  
                X[0] = X[0].tolist();

                single_error = (np.abs(GC_network1.detect(X[0]) - X[0]))   
                
                GC_network2 = AR(self.model.lag *2)
                X[1] = np.array(causality_index2) * np.array(X[1])
                X[1] = X[1].tolist()
                double_error = (np.abs(GC_network2.detect(X)-X[0]))    

        
                single_error_2 = (np.abs(GC_network1.detect(X[1])-X[1]))
        
                single_error  = (single_error[0])[self.model.lag:]
                double_error = (double_error[0])[self.model.lag:]
                single_error_2  = (single_error_2[0])[self.model.lag:]
            
            

                Ftest_win =[]
                for i in range(win_number):  
                    error1_win = 0
                    error2_win = 0

                    for j in range(self.win_len):
                        error1_win += np.abs(single_error[i * self.win_len +j])*np.abs(single_error[i * self.win_len +j])
                        error2_win += np.abs(double_error[i * self.win_len +j])*np.abs(double_error[i * self.win_len +j])

                    Ftest_win.append(error1_win /error2_win)
        
                    print(Ftest_win)
          
        
            #optimize the causal-indexing coefficients
                for i in range(self.model.lag, len(X[0])):
               
                    if int((i-self.model.lag)/self.win_len)!=win_number:
                        '''
                        if Ftest_win[int((i-self.NAR_inputnodes)/self.win_len)] < 0.2:   #increase the recall:
                            if (np.mean(np.tanh(double_error))-np.tanh(double_error[i-self.NAR_inputnodes])) < 0:
                                (causality_index1)[i] = (causality_index1)[i]  + (1-self.index_lr)*(np.mean(np.tanh(double_error))-np.tanh(double_error[i-self.NAR_inputnodes]))
                            #print("causal_index[0]:",(causality_index1))
                            #print("causal_index[0][i-1]:",(causality_index[0])[i-1])
                                (causality_index2)[i] = (causality_index2)[i]  + (1-self.index_lr)*(np.mean(np.tanh(double_error))-np.tanh(double_error[i-self.NAR_inputnodes]))
                        '''
                    
                        if Ftest_win[int((i-self.model.lag)/self.win_len)] > 2:   #increase the accuracy:
                            if (np.mean(np.tanh(single_error))-np.tanh(single_error[i-self.model.lag])) < 0:
                                (causality_index1)[i] = (causality_index1)[i]  + (1-self.index_lr)*(np.mean(np.tanh(single_error))-np.tanh(single_error[i-self.model.lag]))
                       
                                (causality_index2)[i] = (causality_index2)[i]  + (1-self.index_lr)*(np.mean(np.tanh(single_error))-np.tanh(single_error[i-self.model.lag]))    
                        
                    
                count+= 1

            
            
            
        return Ftest_win


