import numpy as np
import matplotlib.pyplot as plt

#loading an array of images
images = np.load('mnist_images1.npy')

#loading an array of labels that correspond to the digit in each image
labels = np.load('mnist_labels1.npy')

#a matrix (number of images (60000), number of pixels for each image (784))
x_all = images.reshape(images.shape[0], 28*28).astype(float) / 255.

#a matrix (60000, 10); for each image it puts a 1 at the digit's index
#Ex: if image has a 5 it will assign 1 to the 5th index.
L_all = np.zeros((len(labels), 10), dtype=int)
L_all[np.arange(len(labels)), labels] = 1

#x0 and L0 are the arrays that contain the last 59000 images and are used for training the program.
x0 = x_all[1000:]
L0 = L_all[1000:]

#x1 and L1 are the arrays that contain the first 1000 images and are used for finding the error
x1 = x_all[:1000]
L1 = L_all[:1000]

error = 100 #Error, initialized at 100

DIGITS = L_all.shape[1]
PIXELS = 784
IMAGES = L0.shape[0]

#a matrix (number of digits (10), number of pixels (784))
A = np.zeros((DIGITS, PIXELS))

#a vector, which contains scalars for the regression
b = np.zeros((DIGITS, 1))

#a matrix, which only contains ones; used for computing one of the gradients
m0 = np.ones((IMAGES, 1))

#a value, which is used to make the gradient smaller in order to avoid divergence
eps = 1e-5

#the regression
def z (A, x, b):
    return A @ x.T + b

#softamx function, used for calculating the probability, whether it's a 0, 1, 2, etc.
def softmax (z):
    return np.exp(-z) / np.exp(-z).sum(0)

#log cost function
def J (L, p):
    return -(L * np.log(p).T).sum() 

#returns the gradients of J with respect to every element in A and every element in b
def grad (L, p, x, m):
    N = -(p - L.T)
    return N @ x, N @ m

#evaluates the error
def evaluate_error(p, L):
    return (1-(p.argmax(0)==L.argmax(1)).mean())*100

i = 0
while True:
    p = softmax(z(A, x0, b))
    p1 = softmax(z(A, x1, b))
    j = J(L0, p)
    g = grad(L0, p, x0, m0)

    #performs the gradient descent
    A -= eps * g[0]
    b -= eps * g[1]

    e = evaluate_error(p1, L1)
    print(j, evaluate_error(p, L0), e)

    #stops the code, when the error begins to increase in order to prevent it from over-training
    if error >= e:
        error = e
        i_last = i
    else:
        if i > i_last + 30:
            break
    i += 1
    
    
