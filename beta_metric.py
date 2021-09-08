# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:27:44 2020

@author: gxjco

This code is used for evaluation based on the beta-VAE metrics
"""
import numpy as np
from sklearn import linear_model

def beta_metric_compute(type_):
 path='C:/Users/gxjco/OneDrive/DESKTOP/graph disentangled VAE/synthetic2_interprete/quantitative_evaluation/'
 z_a1=np.load(path+type_+'_WS_graph_testing_a1_z.npy')
 z_a2=np.load(path+type_+'_WS_graph_testing_a2_z.npy')
 z_b1=np.load(path+type_+'_WS_graph_testing_b1_z.npy')
 z_b2=np.load(path+type_+'_WS_graph_testing_b2_z.npy')
 z_c1=np.load(path+type_+'_WS_graph_testing_c1_z.npy')
 z_c2=np.load(path+type_+'_WS_graph_testing_c2_z.npy')

 z_diff_a=np.mean(np.abs(z_a1-z_a2),axis=1)
 z_diff_b=np.mean(np.abs(z_b1-z_b2),axis=1)
 z_diff_c=np.mean(np.abs(z_c1-z_c2),axis=1)

 #generating the samples
 train_labels=np.concatenate((np.zeros(250),np.ones(250),np.ones(250)*2),axis=0)
 test_labels=np.concatenate((np.zeros(250),np.ones(250),np.ones(250)*2),axis=0)
 train_samples=np.concatenate((z_diff_a[:250],z_diff_b[:250],z_diff_c[:250]),axis=0)[:,:]
 test_samples=np.concatenate((z_diff_a[250:],z_diff_b[250:],z_diff_c[250:]),axis=0)[:,:]

 #train model
 #print("Training sklearn model.")
 model = linear_model.LogisticRegression()
 model.fit(train_samples, train_labels)

 #test model
 #print("Evaluate training set accuracy.")
 train_accuracy = model.score(train_samples, train_labels)
 print(type_+"beta_metric Training set accuracy: %.2g", train_accuracy)
  
 #print("Evaluate testing set accuracy.")
 test_accuracy = model.score(test_samples, test_labels)
 print(type_+"beta_metric Training set accuracy: %.2g", test_accuracy)
 return test_accuracy