import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from sklearn.tree import DecisionTreeClassifier

class Model:

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
        network4 = max_pool_2d(network, kernel_size=1, strides=1)

        network9 = merge([network1, network2, network3, network4] , mode='concat', axis=3)
        network9 = max_pool_2d(network9, kernel_size=2, strides=2)


        network15 = max_pool_2d(network9, kernel_size=1, strides=1)

        network5 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network5 = conv_2d(network5, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        network6 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network6 = conv_2d(network6, 16, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        network7 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network7 = conv_2d(network7, 16, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        network8 = max_pool_2d(network, kernel_size=1, strides=2)

        network10 = merge([network5, network6, network7, network8], mode='concat', axis=3)
        network10 = merge([network9, network10], mode='concat', axis=3)

        network11 = conv_2d(network10, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network11 = conv_2d(network11, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network12 = conv_2d(network10, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network12 = conv_2d(network12, 16, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network13 = conv_2d(network10, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network13 = conv_2d(network13, 16, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        network14 = network = max_pool_2d(network10, kernel_size=1, strides=1)

        network9 = merge([network11, network12, network13, network14], mode='concat', axis=3)
        network9 = merge([network9, network15], mode='concat', axis=3)

        network = max_pool_2d(network9, kernel_size=2, strides=2)


        network = fully_connected(network, 2048, activation='relu')
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, self.output_shape, activation='softmax')


        return network
