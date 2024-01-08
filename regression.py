import numpy as np

x0 = np.asarray([-3, -1, 0, 1, 3.])
y0 = np.asarray([1.5, 2.5, 2, 2.5, 1.5])

kb0 = np.asarray([1., 1.])
eps = 0.01

def f (kb, x):
    return kb[0]*x + kb[1]

def J (kb):
    return ( (y0 - f(kb, x0))**2 ).sum()

def grad (kb):
    return np.asarray( [-2 * ((y0 - f(kb,x0)) * x0).sum(), -2 * (y0 - f(kb,x0)).sum()] )
    

kb = kb0.copy()
j = J(kb)

while abs(j) > 1e-3:
    g = grad(kb)
    kb -= eps * g
    j = J(kb)
    print(kb, j)
    if abs(g).sum() < 1e-7:
       break
