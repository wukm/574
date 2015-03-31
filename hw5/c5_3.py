#!/usr/bin/env python3

from CSData import cs_data
from LASSO import lasso

A, b, x_ex = cs_data(25,500,2)
x_est = lasso(A, b, λ=.0001, tol=.000001)


