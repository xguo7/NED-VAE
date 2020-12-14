from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerVAE
from input_data import *
from model import *
from preprocessing import *

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Settings
for name in list(flags.FLAGS):
      delattr(flags.FLAGS,name)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden11', 10, 'Number of units in hidden layer 11.')
flags.DEFINE_integer('hidden12', 8, 'Number of units in hidden layer 12.')
flags.DEFINE_integer('hidden13', 3, 'Number of units in hidden layer 13.')
flags.DEFINE_integer('hidden21', 10, 'Number of units in hidden layer 21.')
flags.DEFINE_integer('hidden22', 8, 'Number of units in hidden layer 22.')
flags.DEFINE_integer('hidden23', 6, 'Number of units in hidden layer 23.')
flags.DEFINE_integer('hidden24', 3, 'Number of units in linear layer 24.')
flags.DEFINE_integer('hidden31', 10, 'Number of units in hidden layer 31.')
flags.DEFINE_integer('hidden32', 8, 'Number of units in linear layer 32.')
flags.DEFINE_integer('hidden33', 6, 'Number of units in hidden layer 31.')
flags.DEFINE_integer('hidden34', 3, 'Number of units in linear layer 32.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('edge_dropout', 0., 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0., 'Scalar for Graphite')
flags.DEFINE_integer('vae', 1, 'for variational objective')
flags.DEFINE_integer('batch_size', 900, 'Number of samples in a batch.')
flags.DEFINE_integer('decoder_batch_size',900, 'Number of samples in a batch.')
flags.DEFINE_integer('subsample', 0, 'Subsample in optimizer')
flags.DEFINE_float('subsample_frac', 1, 'Ratio of sampled non-edges to edges if using subsampling')
flags.DEFINE_integer('num_feature', 2, 'Number of features.')
flags.DEFINE_integer('verbose', 1, 'Output all epoch data')
flags.DEFINE_integer('test_count', 10, 'batch of tests')
flags.DEFINE_string('model', 'feedback', 'Model string.')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('connected_split', 0, 'use split with training set always connected')
flags.DEFINE_string('type', 'test', 'train or test')
flags.DEFINE_integer('if_visualize', 1, 'varying the z to see the generated graphs')

def ZscoreNormalization(x, mean_, std_):
    """Z-score normaliaztion"""
    x = (x - mean_) / std_
    return x


def main(beta,type_model):
        if 'vae_type' in list(flags.FLAGS):
            delattr(flags.FLAGS,'vae_type')
        flags.DEFINE_string('vae_type', type_model, 'local or global or local_global')   
        if FLAGS.seeded:
            np.random.seed(1)
        
          # Load data
        adj,node= load_data_syn(FLAGS.type)
        adj_orig = adj
        adj_train=adj
        adj_test=adj

        #if FLAGS.features == 0:
         #     feature_test = np.tile(np.identity(adj_test.shape[1]),[adj_test.shape[0],1,1])
          #    feature_train = np.tile(np.identity(adj_train.shape[1]),[adj_train.shape[0],1,1])
              
        #feature_train=features[:]
        #feature_test=features[:]      
            # featureless
        num_nodes = adj.shape[1]
        
        #features = sparse_to_tuple(features.tocoo())
        num_features = node.shape[2]
        pos_weight = float(adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) / adj.sum()
        norm = adj.shape[0] *adj.shape[1] * adj.shape[1] / float((adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) * 2)
        
        adj_orig=adj_train.copy()
        for i in range(adj_train.shape[0]):
            adj_orig[i] = adj_train[i].copy() + np.eye(adj_train.shape[1])
            
        #use encoded label
        adj_label=np.zeros(([adj_train.shape[0],adj_train.shape[1],adj_train.shape[2],2]))  
        for i in range(adj_train.shape[0]):
                for j in range(adj_train.shape[1]):
                    for k in range(adj_train.shape[2]):
                        adj_label[i][j][k][int(adj_orig[i][j][k])]=1
        
        placeholders = {
                'features': tf.compat.v1.placeholder(tf.float32,[FLAGS.batch_size,node.shape[1],node.shape[2],node.shape[3]]),
                'adj': tf.compat.v1.placeholder(tf.float32,[FLAGS.batch_size,adj_train.shape[1],adj_train.shape[2]]),
                'adj_orig': tf.compat.v1.placeholder(tf.float32,[FLAGS.batch_size,adj_train.shape[1],adj_train.shape[2],2]),
                'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
            }
        
        #model = GCNModelFeedback(placeholders, num_features, num_nodes)
        model = GCNModelVAE(placeholders, num_features, num_nodes)
        
       
        if FLAGS.type=='train':
          with tf.name_scope('optimizer'):
                opt = OptimizerVAE(preds_edge=model.rec_edge_logits,
                                   preds_node=model.rec_node,
                                   labels_edge=tf.reshape(placeholders['adj_orig'], [-1,2]),
                                   labels_node=tf.reshape(placeholders['features'], [-1,1]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm,
                                   beta=beta)
        
     
        saver = tf.compat.v1.train.Saver()
        if FLAGS.type=='train':
          with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer()) 
             # Train model
            for epoch in range(FLAGS.epochs):
              batch_num=int(adj_train.shape[0]/FLAGS.batch_size)
              graph=[]
              for i in range(batch_num):
                  adj_batch=adj_train[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  adj_label_batch=adj_label[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  feature_batch=node[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                  t = time.time()
                  # Construct feed dictionary
                  feed_dict = construct_feed_dict(adj_batch, adj_label_batch, feature_batch,placeholders)
                  feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                  # Run single weight update
                  outs = sess.run([opt.opt_op, opt.cost , opt.accuracy], feed_dict=feed_dict)
                  # Compute average loss
                  avg_cost = outs[1]
                  avg_accuracy = outs[2]
                  #if epoch==FLAGS.epochs-1:
                  #    graph.append(outs[3])
        
                  print("Epoch:", '%04d' % (epoch + 1), "disentangle_loss=", "{:.5f}".format(avg_cost-avg_accuracy),
                  "mse_loss=", "{:.5f}".format(avg_accuracy),
                  "time=", "{:.5f}".format(time.time() - t))  
              if epoch>1 and epoch%50==0:
                      save_path = saver.save(sess, "./tmp/model_dgt_global_"+FLAGS.vae_type+".ckpt")
            print("Optimization Finished!")
            print("Model saved in file: %s" % save_path)
            
        def generate_new(adj_test,adj_label,features):
           feed_dict = construct_feed_dict(adj_test, adj_label, features, placeholders)
           feed_dict.update({placeholders['dropout']: 0})
           z_n,z_e,z_g,g,node = sess.run([model.z_mean_n,model.z_mean_e,model.z_mean_g, model.sample_rec_edge,model.sample_rec_node], feed_dict=feed_dict)
           return z_n,z_e,z_g,g,node   
            
        if FLAGS.type=='test':
          with tf.compat.v1.Session() as sess:
            saver.restore(sess, "./tmp/model_dgt_global_"+FLAGS.vae_type+".ckpt")
            print("Model restored.")
            graphs=[]
            nodes=[]
            z_n=[]
            z_e=[]
            z_g=[]
            test_batch_num=1 #12 variables and each variabe has 100 ranging values   #int(adj_train.shape[0]/FLAGS.batch_size)
            for i in range(4,5):#test_batch_num
                adj_batch_test=adj_train[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                adj_batch_label=adj_label[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                feature_batch_test=node[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                z_n_batch,z_e_batch,z_g_batch,g,n=generate_new(adj_batch_test,adj_batch_label,feature_batch_test)
                graphs.append(g)
                nodes.append(n)
                z_n.append(z_n_batch)
                z_e.append(z_e_batch)
                z_g.append(z_g_batch)
            if FLAGS.if_visualize==0:
              graphs=np.array(graphs).reshape(-1,num_nodes,num_nodes)
              nodes=np.array(nodes).reshape(-1,num_nodes,num_features)
              z_n=np.array(z_n)
              z_e=np.array(z_e)
              z_g=np.array(z_g)
              np.save(FLAGS.vae_type+'_z_n.npy',z_n)  
              np.save(FLAGS.vae_type+'_z_e.npy',z_e) 
              np.save(FLAGS.vae_type+'_z_g.npy',z_g) 
            if FLAGS.if_visualize==1:
              np.save('./qualitative_evaluation/'+FLAGS.vae_type+'_graphs_synthetic3.npy',graphs)
              np.save('./qualitative_evaluation/'+FLAGS.vae_type+'_nodes_synthetic3.npy',nodes)
                   
            

if __name__ == '__main__': 
        types=['beta-VAE']   #,,,'InfoVAE',,,   ,'HFVAE'],,,,'InfoVAE''FactorVAE''InfoVAE','DIP-VAE''FactorVAE','HFVAE'
        for t in types:
          tf.compat.v1.reset_default_graph()
          main(10,t)
          #FactorVAE,HFVAE-batch_size is small and lr is small:200,0.00002, 800 epochs, or will easy for gradient clappse