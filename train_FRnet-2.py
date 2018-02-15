X_train = np.reshape(X_train, (-1,64, 64, 1))           # reshaping 4096 into 64*64 with single channel
y_train = np.reshape(y_train, (y_train_len, 1))
X_test = np.reshape(X_test, (-1,64, 64, 1))
y_test = np.reshape(y_test, (y_test_len, 1))



Model = Model(input_shape=[None, 64, 64, 1], output_shape=1477)
network = Model.load_model()
#
network = regression(network, optimizer='adam', learning_rate=0.001, shuffle_batches=True,batch_size=2,
                     loss=categorical_crossentropy, name='target')
#
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit({'input': X_train}, {'target': y_train}, n_epoch=3,batch_size=2,
          validation_set=({'input': X_test}, {'target': y_test}), shuffle=True,
          snapshot_step=100, show_metric=True, run_id='FRnet-2')

model.save("model2.tf")

model = Model.encoded_network

predicted_array = model.predict(X_test)

#
print(predicted_array.shape)
