# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:02:31 2019

@author: gxjco

this code is node edge disentangled VAE model (basement)
"""

from layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class GCNModelVAE(object):
        
    '''VGAE Model for reconstructing graph edges from node representations.'''
    def __init__(self, placeholders, num_features, num_nodes, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)
        self.inputs = placeholders['features']  #node attributes
        self.input_dim = num_features
        #self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']   #edge attributes
        self.dropout = placeholders['dropout']
        self.adj_label = tf.reshape(placeholders['adj_orig'],[-1,2])
        self.weight_norm = 0
        
        self.g_bn_n1 = batch_norm(name='g_bn_n1')
        self.g_bn_n2 = batch_norm(name='g_bn_n2')
 
        
        self.g_bn_e1 = batch_norm(name='g_bn_e1')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        
        self.g_bn_g1 = batch_norm(name='g_bn_g1')
        self.g_bn_g2 = batch_norm(name='g_bn_g2')
        self.g_bn_g3 = batch_norm(name='g_bn_g3')

 
        self.g_bn_e_d0 = batch_norm(name='g_bn_e_d0')
        self.g_bn_e_d1 = batch_norm(name='g_bn_e_d1')
        self.g_bn_e_d2 = batch_norm(name='g_bn_e_d2')
        
        self.g_bn_n_d0 = batch_norm(name='g_bn_n_d0')
        self.g_bn_n_d1 = batch_norm(name='g_bn_n_d1')
            
        self._build()
        
    def _build(self):
        self.encoder()
        self.z_n,self.z_e,self.z_g,z_element = self.get_z(random = True)
        z_noiseless = self.get_z(random = False)
        if not FLAGS.vae:
          z = z_noiseless 
        if FLAGS.type=='train':
          self.rec_edge,self.rec_edge_logits,self.rec_node = self.decoder(self.z_n,self.z_e,self.z_g,z_element)
        if FLAGS.type=='test': 
          self.sample_rec_edge,self.sample_edge_logits,self.sample_rec_node= self.sample()
        
        t_vars = tf.compat.v1.trainable_variables()

        self.vars = [var for var in t_vars]
        self.saver = tf.compat.v1.train.Saver()
        #self.reconstructions_noiseless = self.decoder(z_noiseless)        
        
        
    def encoder(self):
      with tf.compat.v1.variable_scope("encoder") as scope:  
          
        #node attribute embeddings: vector-cnn 
        f1=self.g_bn_n1(n2n(self.inputs, FLAGS.hidden11, k_h=1,name='g_n1_conv'))
        self.dim1=f1.get_shape().as_list()[2] #record the convoled results' length
        f2=self.g_bn_n2(n2n(lrelu(f1), FLAGS.hidden12, k_h=1,name='g_n2_conv'))
        self.dim2=f2.get_shape().as_list()[2]
        self.z_mean_n=linear_en(tf.reshape(f2, [FLAGS.batch_size, -1]), FLAGS.hidden13, 'g_n3_lin')
        self.z_std_n=linear_en(tf.reshape(f2, [FLAGS.batch_size, -1]), FLAGS.hidden13, 'g_n4_lin')
        #batch*l1
        
        
        #edge attributes embeddings: cross-cnn        
        self.adj=tf.reshape(self.adj,[FLAGS.batch_size,self.n_samples,self.n_samples,-1]) 
        e1 = self.g_bn_e1(e2e(lrelu(self.adj), FLAGS.hidden21, k_h=self.n_samples,name='g_e1_conv'))
            # e1 is (n*300 x 300*d )
        e2 = self.g_bn_e2(e2e(lrelu(e1), FLAGS.hidden22, k_h=self.n_samples,name='g_e2_conv'))
        e3=self.g_bn_e3(e2n(lrelu(e2), FLAGS.hidden23,k_h=self.n_samples, name='g_e3_conv'))            
        e3=tf.reshape(e3,[FLAGS.batch_size,self.n_samples,-1])
        self.z_mean_e=linear_en(tf.reshape(e3, [FLAGS.batch_size, -1]), FLAGS.hidden24, 'g_e4_lin')
        self.z_std_e=linear_en(tf.reshape(e3, [FLAGS.batch_size, -1]), FLAGS.hidden24, 'g_e5_lin')        
         #batch*l2
        
        
        #graph entangled embeddings:    
        input_=tf.reshape(self.inputs,[FLAGS.batch_size,self.n_samples,-1])        
        g1=self.g_bn_g1(GraphConvolution_adj(self.adj,input_, FLAGS.hidden31, name='g_g1_conv')) 
        g2=self.g_bn_g2(GraphConvolution_adj(self.adj,lrelu(g1),FLAGS.hidden32, name='g_g2_conv'))            
        #g3=n2g_adj(tf.reshape(g2,[FLAGS.batch_size,self.n_samples,FLAGS.hidden33,1]),1,self.n_samples, name='g_g3_n2g')        
        g3=linear_en(tf.reshape(lrelu(g2), [FLAGS.batch_size, -1]), FLAGS.hidden33, 'g_g3_lin')
        self.z_mean_g=linear_en(tf.reshape(g3, [FLAGS.batch_size, -1]), FLAGS.hidden34, 'g_g4_lin')
        self.z_std_g=linear_en(tf.reshape(g3, [FLAGS.batch_size, -1]), FLAGS.hidden34, 'g_g5_lin')
         #batch*l1

    def get_z(self, random):

        z_n=self.z_mean_n+ tf.random.normal([FLAGS.batch_size,FLAGS.hidden13],stddev=0.1) * tf.exp(self.z_std_n)
        #z_n=tf.reshape(z_n,[FLAGS.batch_size,1,self.z_mean_n.shape[1]])
        
        z_e=self.z_mean_e+ tf.random.normal([FLAGS.batch_size,FLAGS.hidden24],stddev=0.1) * tf.exp(self.z_std_e)
        #z_e=tf.reshape(z_e,[FLAGS.batch_size,1,self.z_mean_e.shape[1]])
        
        z_g= self.z_mean_g+ tf.random.normal([FLAGS.batch_size,FLAGS.hidden34],stddev=0.1) * tf.exp(self.z_std_g)
        
        z_element=tf.random.normal([FLAGS.batch_size,self.n_samples],stddev=0.1)
    
        
        return z_n,z_e,z_g,z_element       
        
        
    def decoder(self, z_n,z_e,z_g,z_element):
        with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            
            #decoing the edge attributes
            z_e_g=tf.concat((z_e,z_g),axis=1) 
            z_e1=linear_de(z_e_g, FLAGS.hidden23, 'e_de_lin1')           
            z_e2=tf.tile(tf.reshape(z_e1,[FLAGS.decoder_batch_size,1,1,-1]),[1,self.n_samples,1,1])            
            z_e3=tf.concat((z_e2,tf.reshape(z_element,[FLAGS.decoder_batch_size,self.n_samples,1,1])),axis=3)
            e_d0_ = self.g_bn_e_d0(z_e3)
                    
            e_d1= de_e2n(tf.nn.relu(e_d0_),
                [FLAGS.decoder_batch_size, self.n_samples, self.n_samples, FLAGS.hidden22],k_h=self.n_samples, name='g_e_d1', with_w=False)           
            e_d1_ = self.g_bn_e_d1(e_d1)
 
            e_d2= de_e2e(tf.nn.relu(e_d1_),
                [FLAGS.decoder_batch_size,self.n_samples, self.n_samples, FLAGS.hidden21],k_h=self.n_samples, name='g_e_d2', with_w=False)            
            e_d2_ = self.g_bn_e_d2(e_d2)
       
            e_d3= de_e2e(tf.nn.relu(e_d2_),
                [FLAGS.decoder_batch_size, self.n_samples, self.n_samples, 2],k_h=self.n_samples, name='g_e_d3', with_w=False)
            diag=np.tile(np.ones(self.n_samples),[FLAGS.decoder_batch_size,1,1])-np.tile(np.eye(self.n_samples),[FLAGS.decoder_batch_size,1,1])
            edge_reconstruction=tf.math.argmax( tf.sigmoid(e_d3),3)*diag 
            edge_logits=tf.reshape(e_d3,[-1,2])
            
            
            #decoding the node attributes:
            z_element=tf.tile(tf.reshape(z_element,[FLAGS.decoder_batch_size,self.n_samples,1,1]),[1,1,self.dim2,1])
            z_n_g=tf.concat((z_n,z_g),axis=1)
            z_n1=linear_de(z_n_g, FLAGS.hidden12*self.dim2, 'n_de_lin1')           
            z_n2=tf.tile(tf.reshape(z_n1,[FLAGS.decoder_batch_size,1,self.dim2,FLAGS.hidden12]),[1,self.n_samples,1,1])            
            z_n3=tf.concat((z_n2,z_element),axis=3)
            n_d0_ = self.g_bn_n_d0(z_n3)
            
            n_d1= de_n2n(tf.nn.relu(n_d0_),
            [FLAGS.decoder_batch_size, self.n_samples,self.dim1, FLAGS.hidden11],k_h=1, name='g_n_d1', with_w=False)           
            n_d1_ = self.g_bn_n_d1(n_d1)
                    
            n_d2= de_n2n(tf.nn.relu(n_d1_),
                [FLAGS.decoder_batch_size, self.n_samples, 2, 1],k_h=1, name='g_n_d2', with_w=False)           
             
            #node_reconstruction=tf.arg_max(tf.sigmoid(n_d2),3)
            node_reconstruction=tf.reshape(n_d2,[FLAGS.decoder_batch_size, self.n_samples, -1])
            
            return   edge_reconstruction,edge_logits, node_reconstruction




    def sample(self):
        if FLAGS.if_visualize==0:
            z_n=self.z_mean_n+ tf.random.normal([FLAGS.batch_size,FLAGS.hidden13],stddev=0.1) * tf.exp(self.z_std_n)       
            z_e=self.z_mean_e+ tf.random.normal([FLAGS.batch_size,FLAGS.hidden24],stddev=0.1) * tf.exp(self.z_std_e)       
            z_g= self.z_mean_g+ tf.random.normal([FLAGS.batch_size,FLAGS.hidden34],stddev=0.1) * tf.exp(self.z_std_g)
            z_element=tf.random.normal([FLAGS.batch_size,self.n_samples],stddev=0.1)
            edge_reconstruction,edge_logits,node_reconstruction= self.decoder(z_n,z_e,z_g,z_element)  
        
            
            
            
        if FLAGS.if_visualize==1:           
            #make one dimension of one node changed and other fixed 
            length=FLAGS.hidden13+FLAGS.hidden24+FLAGS.hidden34
            z_n=np.load(FLAGS.vae_type+'_z_n.npy').reshape(-1,1,FLAGS.hidden13)[1*length:2*length]+ np.random.normal(1,0.1,[length,1,FLAGS.hidden13]) * np.exp(0.1)       
            z_e=np.load(FLAGS.vae_type+'_z_e.npy').reshape(-1,1,FLAGS.hidden24)[1*length:2*length]+ np.random.normal(1,0.1,[length,1,FLAGS.hidden24]) * np.exp(0.1)       
            z_g= np.load(FLAGS.vae_type+'_z_g.npy').reshape(-1,1,FLAGS.hidden34)[1*length:2*length]+ np.random.normal(1,0.1,[length,1,FLAGS.hidden34]) * np.exp(0.1)
            z_element=tf.random.normal([length,1,self.n_samples],stddev=0.1)
            
            z_n=np.tile(z_n,[1,100,1]).reshape(-1,FLAGS.hidden13)
            z_e=np.tile(z_e,[1,100,1]).reshape(-1,FLAGS.hidden24)
            z_g=np.tile(z_g,[1,100,1]).reshape(-1,FLAGS.hidden34)
            z_element=tf.reshape(tf.tile(z_element,[1,100,1]),[-1,self.n_samples])
            
            rang= np.arange(0,20,0.2)
            for fix_dim in range(FLAGS.hidden13): 
              z_n[fix_dim*100:fix_dim*100+100,fix_dim]=rang
            for fix_dim in range(FLAGS.hidden24): 
              base=FLAGS.hidden13*100
              z_e[fix_dim*100+base:fix_dim*100+100+base,fix_dim]=rang      
            for fix_dim in range(FLAGS.hidden34): 
              base=FLAGS.hidden13*100+FLAGS.hidden24*100 
              z_g[fix_dim*100+base:fix_dim*100+100+base,fix_dim]=rang
            edge_reconstruction,edge_logits,node_reconstruction= self.decoder(z_n.astype('float32'),z_e.astype('float32'),z_g.astype('float32'),z_element)
            
        return edge_reconstruction,edge_logits,node_reconstruction    
        
        