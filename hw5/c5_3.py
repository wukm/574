#!/usr/bin/env python3

from CSData import cs_data
from LASSO import lasso
from scipy.linalg import norm

# this is fast, about 7.4ms per loop

A, b, x_ex = cs_data(50,500,4)

#A, b, x_ex = cs_data(5,8,2)

x_est = lasso(A, b, λ=.001, tol=.000001)

error = norm(x_ex - x_est) / norm(x_ex)
print("error is {}".format(error))


