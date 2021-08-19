from matplotlib import pyplot
import numpy
import scipy.linalg

# define a certain line
a, b = .5, -2.
# sample 100 x-locations in range [0,10]
X = numpy.random.random(100) * 10
# compute linear y-values and add some noise
T = a*X + b + numpy.random.normal(0,a/2,X.shape)

# Network output
# x can be a single sample, or several samples
def y(x, w):
    return w[0] + w[1] * x

# Squared loss function
def J(w):
    return numpy.mean((y(X,w) - T) ** 2)

# Compute the gradient at the given w
def gradient(w):
    # iterate over all samples and compute derivatives w.r.t. w_0 and w_1
    return numpy.array((2*numpy.mean(y(X,w) - T),
                        2*numpy.mean((y(X,w) - T) * X)
                        ))

# select initial w-vector; store to compare two methods later
initial_w = numpy.random.random(2) * 2. - 1.
w = initial_w.copy()

# select learning rate; larger values won't do
eta = 0.01
epochs = 0

# compute first gradient
g = gradient(w)
# perform iterative gradient descent
# stopping criterion: small norm of the gradient
while scipy.linalg.norm(g) > 1e-6:
    # do one update step
    w -= eta * g
    # compute new gradient
    g = gradient(w)
    epochs += 1

# print number of epochs and the final loss
# print(initial_w, w)
print(epochs, J(w))

# plot samples, original line and optimal line
pyplot.plot(X, T, "kx")
pyplot.plot([0,10],[b,10*a+b], "r-")
pyplot.plot([0,10],[y(0,w), y(10,w)], "g-")
pyplot.legend(("Data", "Source", "Regressed"), loc="upper left")
pyplot.savefig("Linear.pdf")
pyplot.show()


### adaptive learning rate strategy
# take same initial weight as before
w = initial_w.copy()
epochs = 0

# compute gradient and loss
g = gradient(w)
old_j = J(w)

# perform iterative gradient descent
# with the same stopping criterion
while scipy.linalg.norm(g) > 1e-6:
    # do one update step
    w -= eta * g

    # compute updated loss
    j = J(w)

    # adapt learning rate
    if j >= old_j:
        eta *= 0.5
    else:
        eta *= 1.1

    # compute new gradient and store current loss
    g = gradient(w)
    old_j = j
    epochs += 1

# print number of epochs and the final loss
print(epochs, J(w))