import tensorflow as tf
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def DIP(enc_mean,lambda_od,lambda_d):
            # expectation of mu (mean of distributions)
            exp_mu = tf.reduce_mean(enc_mean, axis=0)
            # expectation of mu mu.tranpose
            mu_expand1 = tf.expand_dims(enc_mean, 1)
            mu_expand2 = tf.expand_dims(enc_mean, 2)
            exp_mu_mu_t = tf.reduce_mean( mu_expand1 * mu_expand2, axis=0)
            # covariance of model mean
            cov = exp_mu_mu_t - tf.expand_dims(exp_mu, 0) * tf.expand_dims(exp_mu, 1)
            diag_part = tf.diag_part(cov)
            off_diag_part = cov - tf.diag(diag_part)
            regulariser_od = lambda_od * tf.reduce_sum(off_diag_part**2)
            regulariser_d = lambda_d * tf.reduce_sum((diag_part - 1)**2)
            dip_vae_regulariser = regulariser_d + regulariser_od
            return dip_vae_regulariser
        
def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(np.math.pi)
  normalization = tf.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean) #[batch_size, batch_size, num_latents]
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)  
      
def total_correlation(z, z_mean, z_logstd):
  """Estimate of total correlation on a batch.
  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)
  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
  Returns:
    Total correlation estimated on a batch.
  """
  z_logvar=tf.log(tf.multiply(tf.exp(z_logstd),tf.exp(z_logstd)))
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product) 

def hierarchical_total_correlation(z1, z1_mean, z1_logstd,z2, z2_mean, z2_logstd,z3, z3_mean, z3_logstd):
  """Estimate of total correlation on a batch.
  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)
  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
  Returns:
    Total correlation estimated on a batch.
  """
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  z1_logvar=tf.log(tf.multiply(tf.exp(z1_logstd),tf.exp(z1_logstd)))
  z2_logvar=tf.log(tf.multiply(tf.exp(z2_logstd),tf.exp(z2_logstd)))
  z3_logvar=tf.log(tf.multiply(tf.exp(z3_logstd),tf.exp(z3_logstd)))
  z=tf.concat((z1,z2,z3),axis=1)
  dim1=z1.shape[1]
  dim2=z2.shape[1]+dim1
  dim3=z3.shape[1]+dim2
  z_mean=tf.concat((z1_mean,z2_mean,z3_mean),axis=1)
  z_logvar=tf.concat((z1_logvar,z2_logvar,z3_logvar),axis=1)
  
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz1 = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob[:,:,0:dim1], axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  log_qz2 = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob[:,:,dim1:dim2], axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  log_qz3 = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob[:,:,dim2:dim3], axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  log_qz_product = log_qz1+log_qz2+log_qz3
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)    
    
        
def KL_div2(mu, sigma, mu1, sigma1):
    '''KL divergence between N(mu,sigma**2) and N(mu1,sigma1**2)'''
    return 0.5 * ((sigma/sigma1)**2 + (mu - mu1)**2/sigma1**2 - 1 + 2*(tf.log(sigma1) - tf.log(sigma)))        
        
class OptimizerVAE(object):
    def __init__(self, preds_edge,preds_node, labels_edge,labels_node, model, num_nodes, pos_weight, norm,beta):

        self.cost=norm*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_edge, targets=labels_edge, pos_weight=pos_weight))
        self.cost+=norm * tf.reduce_mean(tf.math.squared_difference(labels_node,tf.reshape(preds_node,[-1,1])))
        mse_loss=norm*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_edge, targets=labels_edge, pos_weight=pos_weight))+tf.reduce_mean(tf.math.squared_difference(labels_node,tf.reshape(preds_node,[-1,1])))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.log_lik = self.cost
        self.kl_n = (0.5) * tf.reduce_mean(1 + 2 * model.z_std_n - tf.square(model.z_mean_n) - tf.square(tf.exp(model.z_std_n))) #actually -kl
        self.kl_e = (0.5) * tf.reduce_mean(1 + 2 * model.z_std_e - tf.square(model.z_mean_e) - tf.square(tf.exp(model.z_std_e)))
        self.kl_g = (0.5) * tf.reduce_mean(1 + 2 * model.z_std_g - tf.square(model.z_mean_g) - tf.square(tf.exp(model.z_std_g)))
        self.kl=self.kl_n+self.kl_e+self.kl_g  

        if FLAGS.vae_type=='beta-VAE':
            self.cost -=beta*self.kl
            
        if FLAGS.vae_type=='DIP-VAE':
            model.z_mean_g=tf.reshape(model.z_mean_g,[FLAGS.batch_size,-1])
            dip_regulizar=DIP(model.z_mean_n,10,100)+DIP(model.z_mean_e,10,100)+DIP(model.z_mean_g,10,100)
            self.cost-=(self.kl-beta*dip_regulizar)
            
        if FLAGS.vae_type=='InfoVAE':
            model.z_mean_g=tf.reshape(model.z_mean_g,[FLAGS.batch_size,-1])
            dip_regulizar=DIP(model.z_mean_n,10,100)+DIP(model.z_mean_e,10,100)+DIP(model.z_mean_g,10,100)
            self.cost+=beta*dip_regulizar
            
        if FLAGS.vae_type=='VIB':
            self.cost+=gamma*tf.abs(-self.kl-Constant)
            
        if FLAGS.vae_type=='anchorVAE':
            self.cost -=(self.kl+beta*self.kl_e)
            
        if FLAGS.vae_type=='FactorVAE':
            model.z_mean_g=tf.reshape(model.z_mean_g,[FLAGS.batch_size,-1])
            model.z_g=tf.reshape(model.z_g,[FLAGS.batch_size,-1])
            self.cost-=self.kl
            self.cost+=beta*(total_correlation(model.z_n, model.z_mean_n, model.z_std_n)+total_correlation(model.z_e,model.z_mean_e, model.z_std_e)+total_correlation(model.z_g,model.z_mean_g, model.z_std_g))
            
            
        if FLAGS.vae_type=='HFVAE':  
            model.z_mean_g=tf.reshape(model.z_mean_g,[FLAGS.batch_size,-1])
            model.z_g=tf.reshape(model.z_g,[FLAGS.batch_size,-1])
            self.cost-=self.kl
            self.cost+=beta*(total_correlation(model.z_n, model.z_mean_n, model.z_std_n)+total_correlation(model.z_e,model.z_mean_e, model.z_std_e)+total_correlation(model.z_g,model.z_mean_g, model.z_std_g))
            self.cost+=10*hierarchical_total_correlation(model.z_n, model.z_mean_n, model.z_std_n,model.z_e,model.z_mean_e, model.z_std_e,model.z_g,model.z_mean_g, model.z_std_g)   
            

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        #self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_edge, 0.5), tf.int32), tf.cast(labels_edge, tf.int32))
        self.accuracy =mse_loss #tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class Optimizer_classify(object):
    def __init__(self, preds, label, model, num_nodes, pos_weight, norm):

        self.cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=label))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


