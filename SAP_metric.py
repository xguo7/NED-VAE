# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:46:42 2020

@author: gxjco
"""

import numpy as np
from sklearn import svm


def compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
  """Compute score matrix as described in Section 3."""
  num_latents = mus.shape[0]
  num_factors = ys.shape[0]
  score_matrix = np.zeros([num_latents, num_factors])
  for i in range(num_latents):
    for j in range(num_factors):
      mu_i = mus[i, :]
      y_j = ys[j, :]
      if continuous_factors:
        # Attribute is considered continuous.
        cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
        cov_mu_y = cov_mu_i_y_j[0, 1]**2
        var_mu = cov_mu_i_y_j[0, 0]
        var_y = cov_mu_i_y_j[1, 1]
        if var_mu > 1e-12:
          score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
        else:
          score_matrix[i, j] = 0.
      else:
        # Attribute is considered discrete.
        mu_i_test = mus_test[i, :]
        y_j_test = ys_test[j, :]
        classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
        classifier.fit(mu_i[:, np.newaxis], y_j)
        pred = classifier.predict(mu_i_test[:, np.newaxis])
        score_matrix[i, j] = np.mean(pred == y_j_test)
  return score_matrix


def compute_avg_diff_top_two(matrix):
  sorted_matrix = np.sort(matrix, axis=0)
  return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])

def SAP_compute(type_):
  continuous_factors=[True,True,True]
  path='C:/Users/gxjco/OneDrive/DESKTOP/graph disentangled VAE/synthetic2_interprete/quantitative_evaluation/'
  factor=np.transpose(np.load(path+'WS_factor_testing2.npy'))
  code=np.transpose(np.load(path+type_+'_WS_graph_testing2_z.npy').reshape(-1,9))
  train_length=int(factor.shape[1]/2)
  score_matrix = compute_score_matrix(code[:,:train_length], factor[:,:train_length], code[:,train_length:], factor[:,train_length:], continuous_factors)
  # Score matrix should have shape [num_latents, num_factors].

  print(type_+"SAP_score: ",compute_avg_diff_top_two(score_matrix))
  return compute_avg_diff_top_two(score_matrix)
