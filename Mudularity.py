# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:54:07 2020

@author: gxjco
"""
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

def discrete_mutual_info(mus, ys,num_bin):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
          m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :],  make_discretizer(mus[i, :],num_bin))   
  return m

def make_discretizer(target, num_bins):
    """Wrapper that creates discretizers."""
    Dis=KBinsDiscretizer(num_bins, encode='ordinal').fit(target.reshape(-1,1))   
    return Dis.transform(target.reshape(-1,1)).reshape(-1)

def explicitness_per_factor(mus_train, y_train, mus_test, y_test):
  """Compute explicitness score for a factor as ROC-AUC of a classifier.
  Args:
    mus_train: Representation for training, (num_codes, num_points)-np array.
    y_train: Ground truth factors for training, (num_factors, num_points)-np
      array.
    mus_test: Representation for testing, (num_codes, num_points)-np array.
    y_test: Ground truth factors for testing, (num_factors, num_points)-np
      array.
  Returns:
    roc_train: ROC-AUC score of the classifier on training data.
    roc_test: ROC-AUC score of the classifier on testing data.
  """
  x_train = np.transpose(mus_train)
  x_test = np.transpose(mus_test)
  clf = LogisticRegression().fit(x_train, y_train)
  y_pred_train = clf.predict_proba(x_train)
  y_pred_test = clf.predict_proba(x_test)
  mlb = MultiLabelBinarizer()
  roc_train = roc_auc_score(mlb.fit_transform(np.expand_dims(y_train, 1)),
                            y_pred_train)
  roc_test = roc_auc_score(mlb.fit_transform(np.expand_dims(y_test, 1)),
                           y_pred_test)
  return roc_train, roc_test


def modularity(mutual_information):
  """Computes the modularity from mutual information."""
  # Mutual information has shape [num_codes, num_factors].
  squared_mi = np.square(mutual_information)
  max_squared_mi = np.max(squared_mi, axis=1)
  numerator = np.sum(squared_mi, axis=1) - max_squared_mi
  denominator = max_squared_mi * (squared_mi.shape[1] -1.)
  delta = numerator / denominator
  modularity_score = 1. - delta
  index = (max_squared_mi == 0.)
  modularity_score[index] = 0.
  return np.mean(modularity_score)

def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


def modularity_compute(type_):
  num_bin=10
  continuous_factors=[True,True,True]

  path='C:/Users/gxjco/OneDrive/DESKTOP/graph disentangled VAE/synthetic2_interprete/quantitative_evaluation/'
  factor=np.transpose(np.load(path+'WS_factor_testing2.npy'))
  code=np.transpose(np.load(path+type_+'_WS_graph_testing2_z.npy').reshape(-1,9))
  train_length=int(factor.shape[1]/2)
  for i in range(code.shape[0]):
    code[i]=make_discretizer(code[i],num_bin) #make code discrite
  for i in range(factor.shape[0]):
      if continuous_factors[i]==True:
          factor[i]=make_discretizer(factor[i],num_bin) #make the continous factor discrete
        
  mutual_information = discrete_mutual_info(code[:,:train_length], factor[:,:train_length],num_bin)
  # Mutual information should have shape [num_codes, num_factors].
  print(type_+"modularity_score: "+str(modularity(mutual_information)))
  explicitness_score_train = np.zeros([factor[:,:train_length].shape[0], 1])
  explicitness_score_test = np.zeros([factor[:,train_length:].shape[0], 1])
  mus_train_norm, mean_mus, stddev_mus = normalize_data(code[:,:train_length])
  mus_test_norm, _, _ = normalize_data(code[:,train_length:], mean_mus, stddev_mus)
  for i in range(factor[:,:train_length].shape[0]):
      explicitness_score_train[i], explicitness_score_test[i] = \
          explicitness_per_factor(mus_train_norm, factor[:,:train_length][i, :],
                                mus_test_norm, factor[:,train_length:][i, :])
  print(type_+"explicitness_score_train: ",str(np.mean(explicitness_score_train)))
  print(type_+"explicitness_score_test: ",str(np.mean(explicitness_score_test)))
  return modularity(mutual_information),np.mean(explicitness_score_test)


