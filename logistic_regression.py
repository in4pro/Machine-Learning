import numpy as np
import math

x0 = np.asarray([-1, -1, -1, 1, 1, 1.])
y0 = np.asarray([2, 0, -2, 2, 0, -2.])
v0 = np.asarray([x0, y0])
L0 = np.asarray([0, 1, 0, 1, 0, 1])

p = np.asarray([1, 1, 1.])
eps = 1e-2

def z (p, v):
    return (p).sum() * v + p[2]

def G (p, v):
    return (1 / (1 + math.e**(-z(p, v))))

def J (p):
    return (-(1 - L0)*np.log(1 - G(p, v0)) - L0*np.log(G(p, v0))).sum()

def grad (p):
    return np.asarray( [((G(p, v0) - L0) * v0).sum(), ((G(p, v0) - L0) * v0).sum(), (G(p, v0) - L0).sum()] )

while True:
    g = grad(p)
    p -= eps * g 
    j = J(p)
    print(p)
    if abs(g).sum() < 1e-7:
        break
