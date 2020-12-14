from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs



class Graphite(Layer):
    """Graphite layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, batch_size,dropout=0., act=tf.nn.relu, **kwargs):
        super(Graphite, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,input_dim, output_dim]),[batch_size,1,1])    
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs[0]
        recon_1 = inputs[1]
        recon_2 = inputs[2]
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(recon_1, tf.matmul(tf.transpose(recon_1,[0,2,1]), x)) + tf.matmul(recon_2, tf.matmul(tf.transpose(recon_2,[0,2,1]), x))
        outputs = self.act(x)
        return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, batch_size,adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,input_dim, output_dim]),[batch_size,1,1])      
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs
    
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)  
    
    
def GraphConvolution_adj(adj,input_,output_dim, stddev=0.02,
           name="gcn"):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        w_=tf.tile(tf.reshape(w,[1,input_.get_shape()[-1], output_dim]),[input_.get_shape()[0],1,1])      
        new_x=tf.tile(tf.reshape(tf.matmul(input_, w_),[FLAGS.batch_size,1,adj.get_shape()[1],-1]),[1,adj.get_shape()[-1],1,1])
        conv = tf.matmul(tf.transpose(adj,[0,3,1,2]),new_x)
        
        outputs = lrelu(conv)
        #biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        
        return tf.reshape(tf.transpose(outputs,[0,2,1,3]),[FLAGS.batch_size,adj.get_shape()[1],-1])
    
    
class n2g(Layer):
    """aggregate node representation to graph level."""
    def __init__(self,input_dim, batch_size,dropout=0., act=tf.nn.relu, **kwargs):
        super(n2g, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim,20, name="weights")
        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,input_dim,20]),[batch_size,1,1])      
        self.dropout = dropout
        self.act = act
        self.diag=tf.tile(tf.reshape(tf.eye(input_dim),[1,input_dim,input_dim]),[batch_size,1,1])  
        

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.multiply(tf.matmul(self.vars['weights'],x),self.diag)  #only left the diag values 
        outputs = self.act(x)
        return outputs
    
class g2n(Layer):
    """assgin graph representation to node level."""
    def __init__(self,input_dim, batch_size,dropout=0., act=tf.nn.relu, **kwargs):
        super(g2n, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(20,input_dim, name="weights")
        self.vars['weights']=tf.tile(tf.reshape(self.vars['weights'],[1,20,input_dim]),[batch_size,1,1])      
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(self.vars['weights'],x)
        outputs = self.act(x)
        return outputs    



class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        x = tf.transpose(inputs,[0,2,1])
        x = tf.matmul(inputs, x)
        return x
    
def n2n(input_,output_dim,k_h, d_h=1, d_w=1, stddev=0.02,
           name="n2n"):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [1, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')
        bias = tf.compat.v1.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())
        return conv    

def e2e(input_,output_dim,k_h, d_h=1, d_w=1, stddev=0.02,
           name="e2e"):
    with tf.compat.v1.variable_scope(name):
        w1 = tf.compat.v1.get_variable('w1', [1, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv1 = tf.nn.conv2d(input_, w1[0:1,:,:,:], strides=[1, d_h, d_w, 1], padding='VALID')
        biases1 = tf.compat.v1.get_variable('biases1', [output_dim], initializer=tf.constant_initializer(0.0))
        conv1 = tf.reshape(tf.nn.bias_add(conv1, biases1), conv1.get_shape())
        
        #w2 = tf.compat.v1.get_variable('w2', [k_h,k_h, input_.get_shape()[-1], output_dim],
         #                   initializer=tf.truncated_normal_initializer(stddev=stddev))
        #w2=w1
        conv2 = tf.nn.conv2d(input_, tf.transpose(w1,[1,0,2,3]), strides=[1, d_h, d_w, 1], padding='VALID')
        #biases2 = tf.compat.v1.get_variable('biases2', [output_dim], initializer=tf.constant_initializer(0.0))
        #biases2=biases1
        conv2 = tf.reshape(tf.nn.bias_add(conv2, biases1), conv2.get_shape())
        m1 = tf.tile(conv1,[1,1,k_h,1])
        m2 = tf.tile(conv2,[1,k_h,1,1])
        conv = tf.add(m1, m2)
        return conv
    
def e2n(input_,output_dim,k_h=50, d_h=1, d_w=1, stddev=0.02,
           name="e2n"):
     with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [1, k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

        biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv
    
def n2g_adj(input_,output_dim,k_h, stddev=0.02,
           name="n2g"):
     with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [input_.get_shape()[1],1, 1, 1],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
        biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv,w

def de_n2g(input_, output_shape,
             k_h, d_h=1, d_w=1, stddev=0.02,
             name="de_n2g", with_w=False):
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.compat.v1.get_variable('w', [input_.get_shape()[1],1, 1, 1],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w
        else:
            return deconv      

def de_e2n(input_, output_shape,
             k_h, d_h=1, d_w=1, stddev=0.02,
             name="de_e2n", with_w=False):
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w1 = tf.compat.v1.get_variable('w1', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv1 = tf.nn.conv2d_transpose(input_, w1, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases1 = tf.compat.v1.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv1 = tf.reshape(tf.nn.bias_add(deconv1, biases1), deconv1.get_shape())
        
        #w2 = tf.compat.v1.get_variable('w2', [k_h,1,output_shape[-1], input_.get_shape()[-1]],
        #                    initializer=tf.random_normal_initializer(stddev=stddev))
        #w2=tf.transpose(w1,[1,0,2,3])
        deconv2 = tf.nn.conv2d_transpose(tf.transpose(input_,[0,2,1,3]), tf.transpose(w1,[1,0,2,3]), output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        #biases2 = tf.compat.v1.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #biases2=biases1
        deconv2 = tf.reshape(tf.nn.bias_add(deconv2, biases1), deconv1.get_shape())
        deconv=tf.add(deconv1,deconv2)
        
        if with_w:
            return deconv, w1
        else:
            return deconv 
        
def de_n2n(input_, output_shape,
             k_h, d_h=1, d_w=1, stddev=0.02,
             name="de_n2n", with_w=False):
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.compat.v1.get_variable('w', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        biases = tf.compat.v1.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if with_w:
            return deconv, w
        else:
            return deconv        
         
def de_e2e(input_, output_shape,
             k_h=50, d_h=1, d_w=1, stddev=0.02,
             name="de_e2e", with_w=False):
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        input_1=tf.reshape(tf.reduce_sum(input_,axis=1),(int(input_.shape[0]),k_h,1,int(input_.shape[3]))) 
        input_2=tf.reshape(tf.reduce_sum(input_,axis=2),(int(input_.shape[0]),1,k_h,int(input_.shape[3]))) 
        
        w1 = tf.compat.v1.get_variable('w1', [1,k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv1 = tf.nn.conv2d_transpose(input_1, w1, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')       
        biases1 = tf.compat.v1.get_variable('biases1', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv1 = tf.reshape(tf.nn.bias_add(deconv1, biases1), deconv1.get_shape())

        #w2 = tf.compat.v1.get_variable('w2', [k_h,1, output_shape[-1], input_.get_shape()[-1]],
         #                   initializer=tf.random_normal_initializer(stddev=stddev))
        #w2=tf.transpose(w1,[1,0,2,3])
        deconv2 = tf.nn.conv2d_transpose(input_2, tf.transpose(w1,[1,0,2,3]), output_shape=output_shape,
                                strides=[1, d_h, d_w, 1],padding='VALID')
        #biases2 = tf.compat.v1.get_variable('biases2', [output_shape[-1]], initializer=tf.constant_initializer(0.0))  
        #biases2=biases1
        deconv2 = tf.reshape(tf.nn.bias_add(deconv2, biases1), deconv2.get_shape())
        
        deconv=tf.add(deconv1,deconv2)/2
        if with_w:
            return deconv, w1
        else:
            return deconv  
        
def linear_en(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
def linear_de(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias        

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)
        