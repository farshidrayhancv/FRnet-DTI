from __future__ import division, print_function, absolute_import
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, Normalizer
import numpy as np
import tflearn
from tflearn.objectives import binary_crossentropy,categorical_crossentropy
from tflearn.layers.estimator import regression
from Model import Model
import pandas as pd


dataset = 'dataset.csv'   # each row contails: A dummy value + 1476 values + Class  
df = pd.read_csv(dataset, header=None)
df['label'] = df[df.shape[1] - 1]
df.drop([df.shape[1] - 2], axis=1, inplace=True)
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])


normalization_object = Normalizer()
X = normalization_object.fit_transform(X)
number_of_split = 5
skf = StratifiedKFold(n_splits=number_of_split, shuffle=True)

X_train = None
X_test = None
y_train = None
y_test = None
y_train_len = 0
y_test_len = 0

for train_index, test_index in skf.split(X, y):
    X_train = X[train_index]
    y_test = y[test_index]
    X_test = X[test_index]

    y_train = y[train_index]
    
    y_train_len = len(y_train)
    y_test_len = len(y_test)

    break


X_train = np.reshape(X_train, (-1,211, 7, 1))
y_train = np.reshape(y_train, (y_train_len, 1))
X_test = np.reshape(X_test, (-1,211, 7, 1))
y_test = np.reshape(y_test, (y_test_len, 1))



Model = Model(input_shape=[None, 211, 7, 1], output_shape=1477)
network = Model.load_model()

network = regression(network, optimizer='adam', learning_rate=0.001, shuffle_batches=True,batch_size=2,
                     loss=categorical_crossentropy, name='target')

model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit({'input': X_train}, {'target': np.reshape(X_train, (-1,1477))}, n_epoch=3,batch_size=2,
          validation_set=({'input': X_test}, {'target': np.reshape(X_test, (-1,1477))}), shuffle=True,
          snapshot_step=100, show_metric=True, run_id='FRnet-1')


# model.save("model2.tf")   # Save model optional  

model = Model.encoded_network   # loading the model that outputs 4096 features for each instance 

predicted_features = model.predict(X_test)
