# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:21:13 2020

@author: gxjco
"""
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import KBinsDiscretizer

def discrete_mutual_info(mus, ys,num_bin):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
          m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :],  make_discretizer(mus[i, :],num_bin))   
  return m

def discrete_entropy(ys,num_bin):
  """Compute discrete entropy of the factors."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(make_discretizer(ys[j, :],num_bin),make_discretizer(ys[j, :],num_bin))
  return h

def make_discretizer(target, num_bins):
    """Wrapper that creates discretizers."""
    Dis=KBinsDiscretizer(num_bins, encode='ordinal').fit(target.reshape(-1,1))
    
    return Dis.transform(target.reshape(-1,1)).reshape(-1)

def MIG_compute(type_):
  num_bin=10
  path='C:/Users/gxjco/OneDrive/DESKTOP/graph disentangled VAE/synthetic2_interprete/quantitative_evaluation/'
  factor=np.transpose(np.load(path+'WS_factor_testing2.npy'))
  code=np.transpose(np.load(path+type_+'_WS_graph_testing2_z.npy').reshape(-1,9))

  m = discrete_mutual_info(code, factor,num_bin)
  # m is [num_latents, num_factors]
  entropy_h = discrete_entropy(factor,num_bin)
  sorted_m = np.sort(m, axis=0)[::-1]
  print('MIG score: '+str(np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy_h[:]))))
  return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy_h[:]))