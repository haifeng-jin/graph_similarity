import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np


def layer1(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)  # theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = T.tanh(m)
    return h


def layer2(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)  # theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    return h


def grad_desc(cost, theta):
    alpha = 0.1  # learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))


theta1 = theano.shared(np.array(np.random.rand(3, 3), dtype=theano.config.floatX))  # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(4, 1), dtype=theano.config.floatX))
x = T.dvector()
y = T.dscalar()

hid1 = layer1(x, theta1)  # hidden layer

out1 = T.sum(layer2(hid1, theta2))  # output layer
fc = (out1 - y) ** 2  # cost expression
cost = theano.function(inputs=[x, y], outputs=fc, updates=[
    (theta1, grad_desc(fc, theta1)),
    (theta2, grad_desc(fc, theta2))])
run_forward = theano.function(inputs=[x], outputs=out1)

inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]]).reshape(4, 2)  # training data X
exp_y = np.array([1, 1, 0, 0])  # training data Y
cur_cost = 0
for i in range(1000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k])  # call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0:  # only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

# Training done! Let's test it out
print(run_forward([0, 1]))
print(run_forward([1, 1]))
print(run_forward([1, 0]))
print(run_forward([0, 0]))
