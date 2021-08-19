import numpy
import scipy.linalg
from matplotlib import pyplot

def logistic_fun(a):
    return 1/(1+numpy.exp(-1*a))

# x is [N x 1], W1 is [k x 2], H is [k+1 x N]
def hidden_neurons_output(x, W1):
    A = numpy.dot(W1,numpy.vstack((numpy.ones(x.size),x)))
    H_ = logistic_fun(A)
    H = numpy.vstack((numpy.ones(x.size),H_))
    return H

# w2 is [k+1 x 1], Y is [N x 1]
def network_output(x, W1, w2):
    H = hidden_neurons_output(x, W1)
    Y = numpy.dot(w2.transpose(),H).transpose()
    return H, Y

# Squared loss function
# t is [N x 1]
def J(x, t, W1, w2):
    return numpy.mean((network_output(x, W1, w2)[1] - t) ** 2)

# Compute the gradient at the given W1 and w2
# grad_W1 is [k x 2], grad_w2 is [k+1 x 1]
def gradient(x, t, W1, w2):
    H, Y = network_output(x, W1, w2)

    grad_w2 = 2/x.size*numpy.dot(H,Y - t)

    tmp = numpy.dot(numpy.outer(w2,(Y - t))*(H*(1-H)),
                    numpy.vstack((numpy.ones(x.size),x)).transpose())
    tmp = tmp[1:,:]
    grad_W1 = 2/x.size*tmp

    return grad_W1, grad_w2

# Directly compute the norm of the gradient
# perform gradient descent with implemented adaptive learning rate strategy
def gradient_descent(x, t, W1, w2, eta):
    j_arr = []
    j = J(x, t, W1, w2)
    # stopping criterion part 1: limit the number of iterations
    for i in range(200000):
        # compute gradient for current W1 and w2
        grad_W1, grad_w2 = gradient(x, t, W1, w2)
        g = numpy.hstack((grad_W1.flatten(), grad_w2))
        j_arr.append(j)
        old_j = j
        # stopping criterion part 2: if norm of gradient is small, stop
        if scipy.linalg.norm(g) < 1e-4:
            break
        # perform one gradient descent step
        W1 -= eta * grad_W1
        w2 -= eta * grad_w2

        # compute updated loss
        j = J(x,t,W1,w2)
        # adapt learning rate
        if j >= old_j:
            eta *= 0.5
        else:
            eta *= 1.1

        if (i+1)%10000==0:
            print(f"Epochs: {i+1}")

    # return the optimized weight and number of epochs
    return W1, w2, i+1, j_arr

def fun_1(x):
    return (numpy.cos(3*x) + 1)/2

def fun_2(x):
    return numpy.exp(-0.25*x**2)

n_12_20 = numpy.random.random(20) * 4. - 2.
n_12_1000 = numpy.random.random(1000) * 4. - 2.

def fun_3(x):
    return (x**5 + 3*x**4 - 11*x**3 - 27*x**2 + 10*x + 64)/100

n_3_20 = numpy.random.random(20) * 8. - 4.5
n_3_1000 = numpy.random.random(1000) * 8. - 4.5

# select learning rate
eta = 0.01

# # select number of hidden neurons for fun_1(x)
# k = 3
# # select initial W1 and w2
# initial_W1 = numpy.random.random((k,2)) * 2. - 1.
# initial_w2 = numpy.random.random(k+1) * 2. - 1.
#
# W1, w2, epochs, j_arr = gradient_descent(n_12_20, fun_1(n_12_20), initial_W1, initial_w2, eta)
#
# # plot training data, original function and approximated function
# pyplot.plot(n_12_20, fun_1(n_12_20), "kx")
# pyplot.plot(numpy.linspace(-2,2.01,1000),fun_1(numpy.linspace(-2,2.01,1000)), "r-")
# pyplot.plot(numpy.linspace(-2,2.01,1000),network_output(numpy.linspace(-2,2.01,1000), W1, w2)[1], "g-")
# pyplot.legend(("Training data", "Original fun", "Approximated fun"), loc="upper left")
# pyplot.savefig("Approx_fun1.pdf")
# pyplot.show()
# pyplot.plot(j_arr, "b-")
# pyplot.legend(("Loss"), loc="upper right")
# pyplot.savefig("LossProgression_fun1.pdf")
# pyplot.show()

# # select number of hidden neurons for fun_2(x)
# k = 2
# # select initial W1 and w2
# initial_W1 = numpy.random.random((k,2)) * 2. - 1.
# initial_w2 = numpy.random.random(k+1) * 2. - 1.
#
# W1, w2, epochs, j_arr = gradient_descent(n_12_20, fun_2(n_12_20), initial_W1, initial_w2, eta)
#
# # plot training data, original function and approximated function
# pyplot.plot(n_12_20, fun_2(n_12_20), "kx")
# pyplot.plot(numpy.linspace(-2,2.01,1000),fun_2(numpy.linspace(-2,2.01,1000)), "r-")
# pyplot.plot(numpy.linspace(-2,2.01,1000),network_output(numpy.linspace(-2,2.01,1000), W1, w2)[1], "g-")
# pyplot.legend(("Training data", "Original fun", "Approximated fun"), loc="upper left")
# pyplot.savefig("Approx_fun2.pdf")
# pyplot.show()
# pyplot.plot(j_arr, "b-")
# pyplot.legend(("Loss"), loc="upper right")
# pyplot.savefig("LossProgression_fun2.pdf")
# pyplot.show()

# select number of hidden neurons for fun_3(x)
k = 5
# select initial W1 and w2
initial_W1 = numpy.random.random((k,2)) * 2. - 1.
initial_w2 = numpy.random.random(k+1) * 2. - 1.

W1, w2, epochs, j_arr = gradient_descent(n_3_20, fun_3(n_3_20), initial_W1, initial_w2, eta)

# plot training data, original function and approximated function
pyplot.plot(n_3_20, fun_3(n_3_20), "kx")
pyplot.plot(numpy.linspace(-4.5,3.51,1000),fun_3(numpy.linspace(-4.5,3.51,1000)), "r-")
pyplot.plot(numpy.linspace(-4.5,3.51,1000),network_output(numpy.linspace(-4.5,3.51,1000), W1, w2)[1], "g-")
pyplot.legend(("Training data", "Original fun", "Approximated fun"), loc="upper left")
pyplot.savefig("Approx_fun3.pdf")
pyplot.show()
pyplot.plot(j_arr, "b-")
pyplot.legend(("Loss"), loc="upper right")
pyplot.savefig("LossProgression_fun3.pdf")
pyplot.show()
