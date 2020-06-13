import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import PIL
from PIL import Image
import time
import os

# temporary dataset2 code and info
# imageset2directory = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\pokemon-a\\"
# directory2size = 300*300
# images2 = []
# #only goes up to volcanion, not gen 7 or 8
# for i in range(1, 721):
#     image = Image.open(imageset2directory+str(i)+'.png')
#     images2.append(image)
# placeholder2 = tf.placeholder(tf.float32, shape=([None, directory2size]))


#represent the strength of connections between units.
def weight_variable(shape, name):
    #Outputs random values from a truncated normal distribution.
    #truncated means the value is either bounded below or above (or both)
    initial = tf.truncated_normal(shape, stddev=0.1)
    #A Variable is a modifiable tensor that lives in TensorFlow’s graph of 
    #interacting operations. It can be used and even modified by the computation. 
    #For machine learning applications, one generally has the model parameters 
    #be Variables.
    return tf.Variable(initial, name=name)

#Bias nodes are added to increase the flexibility of 
#the model to fit the data. Specifically, it allows the 
#network to fit the data when all input features are equal to 00, 
#and very likely decreases the bias of the fitted values elsewhere in the data space
def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

#Neurons in a fully connected layer have full connections to 
#all activations in the previous layer, as seen in regular Neural Networks. 
#Their activations can hence be computed with a matrix multiplication followed by a 
#bias offset. 
def FC_layer(X, W, b):
    return tf.matmul(X, W) + b

def main():
    df = pd.read_csv(r'C:\Users\dswhi\.vscode\Pokemon Image Creator\pokemon_list.csv', encoding='latin-1')
    imageset1directory = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\MyScrapedImages\\"
    images1 = []
    for pokemon in df['name']:
        image = Image.open(imageset1directory+pokemon+'.png')
        images1.append(image)
    train = images1[0:round(.8*len(images1))]
    test = images1[round(.8*len(images1)):]
    directory1size = 475*475
    placeholder1 = tf.placeholder(tf.float32, shape=([None, directory1size]))
    latent_dim = 20
    num_neurons = 500
    #layer 1
    W_enc = weight_variable([directory1size, num_neurons], 'W_enc')
    b_enc = bias_variable([num_neurons], 'b_enc')
    #A rescaling of the logistic sigmoid, such that its outputs range from -1 to 1.
    #avoids vanishing gradient problem
    h_enc = tf.nn.tanh(FC_layer(placeholder1, W_enc, b_enc))

    #layer 2- mean
    W_mu = weight_variable([num_neurons, latent_dim], 'W_mu')
    b_mu = bias_variable([latent_dim], 'b_mu')
    mu = FC_layer(h_enc, W_mu, b_mu)

    #generate a vector of means and a vector of standard deviations for reparamterization instead of real values
    W_logstd = weight_variable([num_neurons, latent_dim], 'W_logstd')
    b_logstd = bias_variable([latent_dim], 'b_logstd')
    logstd = FC_layer(h_enc, W_logstd, b_logstd)

    #randomness factor
    noise = tf.random_normal([1, latent_dim])
    #sample from the standard deviations (tf.exp computes exponential of x element-wise) 
    #this is our latent variable we will pass to the decoder
    z = mu + tf.multiply(noise, tf.exp(.5*logstd))
    #The greater standard deviation on the noise added, the less information we can pass using that one variable.
    #encode better early on so that the end result is better from randomly generated latent variables

    #DECODING
    #layer 1
    W_dec = weight_variable([latent_dim, num_neurons], 'W_dec')
    b_dec = bias_variable([num_neurons], 'b_dec')
    h_dec = tf.nn.tanh(FC_layer(z, W_dec, b_dec))

    #layer 2 with a measurement of original images
    #bernouli
    W_rec = weight_variable([num_neurons, directory1size], 'W_rec')
    b_rec = bias_variable([directory1size], 'b_rec')
    rec = tf.nn.sigmoid(FC_layer(h_dec, W_rec, b_rec))

    #loss function
    #Measures how effectively the decoder reconstrcuts an input given its latent representation
    log_likelihood = tf.reduce_sum(placeholder1*tf.log(rec + 1e-9)+(1 - placeholder1)
    *tf.log(1 - rec + 1e-9), reduction_indices=1)
    #subtracts a value from the score depending on how wrong it is
    #minimizes the distance between two points by minimizing score
    KL_term = -.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu,2) - tf.exp(2*logstd), reduction_indices=1)
    # This allows us to use stochastic gradient descent with respect to the variational parameters
    variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
    optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)
    

    #TAKES TOO MUCH MEMORY UP TO TRAIN
    # init = tf.global_variables_initializer()
    # sess = tf.InteractiveSession()
    # sess.run(init)
    # saver = tf.train.Saver()

    # num_iterations = 100000
    # recording_interval = 1000
    # #store value for these 3 terms so we can plot them later
    # variational_lower_bound_array = []
    # log_likelihood_array = []
    # KL_term_array = []

    # iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]
    # for i in range(num_iterations):
    #     # np.round to make MNIST binary
    #     #get first batch (200 digits)
    #     x_batch = np.round(train.next_batch(150)[0])
    #     #run our optimizer on our data
    #     sess.run(optimizer, feed_dict={placeholder1: x_batch})
    #     if (i%recording_interval == 0):
    #         #every 1K iterations record these values
    #         vlb_eval = variational_lower_bound.eval(feed_dict={placeholder1: x_batch})
    #         print("Iteration: {}, Loss: {}".format(i, vlb_eval))
    #         variational_lower_bound_array.append(vlb_eval)
    #         log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={placeholder1: x_batch})))
    #         KL_term_array.append(np.mean(KL_term.eval(feed_dict={placeholder1: x_batch})))
    # load_model = False
    # if load_model:
    #     saver.restore(sess, os.path.join(os.getcwd(), "Trained Bernoulli VAE"))
    # num_pairs = 2
    # image_indices = np.random.randint(0, 150, num_pairs)
    # for pair in range(num_pairs):
    #     #reshaping to show original test image
    #     x = np.reshape(test[image_indices[pair]], (1,directory1size))
    #     plt.figure()
    #     x_image = np.reshape(x, (28,28))
    #     plt.subplot(121)
    #     plt.imshow(x_image)
    #     #reconstructed image, feed the test image to the decoder
    #     x_reconstruction = rec.eval(feed_dict={placeholder1: x})
    #     #reshape it to original size pixels
    #     x_reconstruction_image = (np.reshape(x_reconstruction, (475,475)))
    #     #plot it!
    #     plt.subplot(122)
    #     plt.imshow(x_reconstruction_image)


if __name__=="__main__":
    main()