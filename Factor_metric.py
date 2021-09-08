# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:19:55 2020

@author: gxjco
"""

import numpy as np
from sklearn import linear_model

def prune_dims(variances, threshold=0.005):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold

def factor_metric_compute(type_):
  path='C:/Users/gxjco/OneDrive/DESKTOP/graph disentangled VAE/synthetic2_interprete/quantitative_evaluation/'
  z_a1=np.load(path+type_+'_WS_graph_testing_a1_z.npy')
  z_a2=np.load(path+type_+'_WS_graph_testing_a2_z.npy')
  z_b1=np.load(path+type_+'_WS_graph_testing_b1_z.npy')
  z_b2=np.load(path+type_+'_WS_graph_testing_b2_z.npy')
  z_c1=np.load(path+type_+'_WS_graph_testing_c1_z.npy')
  z_c2=np.load(path+type_+'_WS_graph_testing_c2_z.npy')

  D=z_a1.shape[2]
  L=3 #3 factors

  train_labels=np.concatenate((np.zeros(250),np.ones(250),np.ones(250)*2),axis=0)
  test_labels=np.concatenate((np.zeros(250),np.ones(250),np.ones(250)*2),axis=0)
  train_samples=np.concatenate((z_a1[:250],z_b1[:250],z_c1[:250]),axis=0)[:,:]
  test_samples=np.concatenate((z_a1[250:],z_b1[250:],z_c1[250:]),axis=0)[:,:]

  global_var=np.var(train_samples.reshape([-1,9]),axis=0)
  active_dim=prune_dims(global_var)

  #generate each batch
  for i in range(500):
    train_samples[i,:,:]/=np.std(train_samples[i,:,:])
    test_samples[i,:,:]/=np.std(test_samples[i,:,:])
    
  train_var=np.var(train_samples,axis=1)
  test_var=np.var(test_samples,axis=1)
  train_index=np.zeros(len(train_var))
  test_index=np.zeros(len(test_var))
  
  for i in range(len(train_var)):
    train_index[i]=np.argmin(train_var[i][active_dim])
    test_index[i]=np.argmin(test_var[i][active_dim])

  #votes
  training_votes=np.zeros((D,L))
  testing_votes=np.zeros((D,L))
  for i in range(len(train_index)):
      training_votes[int(train_index[i]),int(train_labels[i])]+=1
      testing_votes[int(test_index[i]),int(test_labels[i])]+=1
    
  #classifier    
  C=np.argmax(training_votes,axis=1)
  other_index = np.arange(training_votes.shape[0])

  #evaluate
  train_accuracy = np.sum(training_votes[other_index,C]) * 1. / np.sum(training_votes)
  test_accuracy = np.sum(testing_votes[other_index,C]) * 1. / np.sum(testing_votes)
  print(type_+"factor Training set accuracy: ", train_accuracy)
  print(type_+"factor Evaluation set accuracy: ", test_accuracy)
  return test_accuracy








