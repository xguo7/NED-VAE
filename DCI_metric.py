# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:19:55 2020

@author: gxjco
"""
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import scipy
from sklearn.preprocessing import KBinsDiscretizer


def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)

def make_discretizer(target, num_bins):
    """Wrapper that creates discretizers."""
    Dis=KBinsDiscretizer(num_bins, encode='ordinal').fit(target.reshape(-1,1))   
    return Dis.transform(target.reshape(-1,1)).reshape(-1)

def DCI_metric_compute(type_):
  num_bin=10
  continuous_factors=[True,True,True]
  path='C:/Users/gxjco/OneDrive/DESKTOP/graph disentangled VAE/synthetic2_interprete/quantitative_evaluation/'
  factor=np.transpose(np.load(path+'WS_factor_testing2.npy'))
  code=np.transpose(np.load(path+type_+'_WS_graph_testing2_z.npy').reshape(-1,9))
  for i in range(factor.shape[0]):
    if continuous_factors[i]==True:
        factor[i]=make_discretizer(factor[i],num_bin) #make the continous factor as discrete
  train_length=int(factor.shape[1]/2)

  importance_matrix, train_err, test_err = compute_importance_gbt(
      code[:,:train_length], factor[:,:train_length], code[:,train_length:], factor[:,train_length:])
  print(type_+"DCI informativeness_train: "+str(train_err))
  print(type_+"DCI informativeness_test： "+str(test_err))
  print(type_+"DCI disentanglement:"+str(disentanglement(importance_matrix)))
  print(type_+"DCI completeness: "+str(completeness(importance_matrix)))
  return test_err, disentanglement(importance_matrix),completeness(importance_matrix)