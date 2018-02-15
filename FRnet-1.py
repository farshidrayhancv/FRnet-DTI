import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge

class Model:
    # Building convolutional network

    input_shape = None
    output_shape = None
    encoded_network = None

    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def load_model(self):

        network = input_data(shape=self.input_shape, name='input')
        network = conv_2d(network, 32, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        network = max_pool_2d(network, kernel_size=2, strides=2)

        network1 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network1 = conv_2d(network1, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network2 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network2 = conv_2d(network2,16, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network3 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network3 = conv_2d(network3, 16, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network4 = network = max_pool_2d(network, kernel_size=1, strides=1)

        network = merge([network1, network2, network3, network4] , mode='concat', axis=3)

        print('final network :', network)

        network = fully_connected(network, 4096, activation='relu')
        self.encoded_network = network
        network = fully_connected(network, 2048, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, self.output_shape, activation='softmax')


        return network

    def get_encoded_network(self):
        return self.encoded_network
