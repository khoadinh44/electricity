import tflearn

def network()
  # Build neural network
  net = tflearn.input_data(shape=[None, 1])
  net = tflearn.fully_connected(net, 32)
  net = tflearn.fully_connected(net, 32)
  net = tflearn.fully_connected(net, 2, activation='softmax')
  net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
  return net
