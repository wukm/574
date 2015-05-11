import numpy

# problem 1a
def g(x):

    x1, x2 = x
    
    term_1 = 3*x1**2 - x2**2
    term_2 = 3*x1*x2**2 - x1**3 - 1

    return term_1**2 + term_2**2

def grad(x):
    
    x1, x2 = x

    # original g
    term_1 = 3*x1**2 - x2**2
    term_2 = 3*x1*x2**2 - x1**3 - 1

    g1 = 2*term_1*(6*x1) + 2*term_2*(3*x2**2 - 3*x1**2)
    g2 = 2*term_1*(-2*x2) + 2*term_2*(6*x1*x2)

    return numpy.array([g1,g2])

def steepest_descent(x, tol=.05, maxit=20000):
    """
    returns an array x, final function value, and number of iterations
    """
    
    for i in range(maxit):

        g1 = g(x)
        z = grad(x)
        z0 = numpy.linalg.norm(z)

        if z0 == 0:
            print("gradient is zero")
            return x, g1, i+1
    
        z /= z0

        # do line search
        a1 = 0
        a3 = 1
        g3 = g(x - a3*z)

        while g3 >= g1:
            a3 /= 2
            g3 = g(x - a3*z)

            if a3 < tol/2:
                print("step sizes getting very small, by min?")
                return x, g1, i+1

        a2 = a3/2
        g2 = g(x-a2*z)

        h1 = (g2 - g1) / a2
        h2 = (g3 - g2) / (a3 - a2)
        h3 = (h2 - h1) / a3

        a0 = .5*(a2 - h1/h3)
        g0 = g(x-a0*z)

        if g0 < g3:
            x -= a0*z
            gg = g0
        else:
            x -= a3*z
            gg = g3
        if abs(gg) < tol:
            print("got to tolerance")
            return x, gg, i+1

        # debug:

        print("{}th iteration, |grad|={}, x={}".format(i, z0, x))
    else:
        #if no break
        print("maxed out iterations")
        return x, gg, i+1


if __name__ == "__main__":
    
    x0 = numpy.array([2.,2.])

    x_end, g_end, it = steepest_descent(x0, tol=.05, maxit=2000)

    print('*'*80)
    print('ending x:', x_end)
    print('g(x)=', g(x_end))
