import numpy as np
import matplotlib.pyplot as plt

#loading an array of images
images = np.load('mnist_images.npy')

#loading an array of labels that correspond to the digit in each image
L_all = np.load('mnist_labels.npy')

#a matrix (number of images (12665), number of pixels for each image (784))
x_all = images.reshape(images.shape[0], 28*28).astype(float) / 255.

#x0 and L0 are the arrays that contain the last 11665 images and are used for training the program.
x0 = x_all[1000:]
L0 = L_all[1000:]

#x1 and L1 are the arrays that contain the first 1000 images and are used for finding the error
x1 = x_all[:1000]
L1 = L_all[:1000]
error = 100 #Error, initialized at 100

#array of parameters; for now contains only zeroes
p = np.zeros((785,))-.01

#a value, which is used to make the gradient smaller in order to avoid divergence
eps = 1e-4

#the regression
def z (p, x):
    return (p[np.newaxis,0:784] * x).sum(axis=1) + p[784]

#sigmoid function, used for calculating the probability, whether it's a 0 or a 1
def sigmoid (y):
    return (1 / (1 + np.exp(-y)))

#log cost function
def J (p):
    return (-(1 - L0)*np.log(1 - sigmoid(z(p,x0))) - L0*np.log(sigmoid(z(p,x0)))).sum(axis=0)

#gradients of J with respect to each parameter and with the respect to the scalar (last parameter)
def grad (p):
    #sum of all the gradients with respect to each parameter
    grad_p01 = ( (sigmoid(z(p,x0)) - L0)[:,np.newaxis] * x0 ).sum(axis=0)
    
    #sums of all the gradients with respect to the scalar
    grad_p2 = ( (sigmoid(z(p,x0)) - L0) ).sum(axis=0)
    return np.hstack( [grad_p01, grad_p2] )

#evaluates the error
def evaluate_error(p, x, L):
    return 1 - ( (z(p, x) > 0) == L ).mean()

i = 0
while True:
    j = J(p)
    g = grad(p)

    #performs the gradient descent
    p -= eps * g
    
    e = evaluate_error(p, x1, L1)
    print(j, evaluate_error(p, x0, L0), e)

    #stops the code, when the error begins to increase in order to prevent it from over-training
    if error >= e:
        error = e
        i_last = i
    else:
        if i > i_last + 30:
            break
    i += 1
