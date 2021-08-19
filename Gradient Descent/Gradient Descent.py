import matplotlib
# enable interactive plot feature in PyCharm. Do not forget to reduce generated plots to one
# matplotlib.use('Qt5Agg')
# matplotlib.rcParams["text.usetex"] = True
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

import numpy
import scipy.linalg

# function to compute the loss for the given w1 and w2
def compute_loss(w1, w2):
    return w1**2 + w2**2 + 30. * numpy.sin(w1) * numpy.sin(w2)

# surface plot of the loss function
def plot_surface(alpha=.8):
    # define range of data samples
    w = numpy.arange(-10, 10.001, 0.1)
    w1, w2 = numpy.meshgrid(w,w)
    # compute the loss for all values of w1 and w2
    J = compute_loss(w1, w2)
    # initialize 3D plot
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection="3d", azim = -40, elev = 50)
    # plot surface with jet colormap
    ax.plot_surface(w1, w2, J, cmap="jet", alpha=alpha)
    return fig, ax

# loss function:
# w_1^2 + w_2^2 + 30 * sin(w_1) * sin(w_2)
# derivatives:
# - w_1: 2w_1 + 30cos(w_1)sin(w_2)
# - w_2: 2w_2 + 30cos(w_2)sin(w_1)
def compute_gradient(w):
    return 2.*w + 30. * numpy.cos(w) * numpy.sin(w[::-1])

# perform gradient descent from the given
def gradient_descent(w, eta):
    # stopping criterion part 1: limit the number of iterations
    for j in range(1000):
        # compute gradient for current w
        g = compute_gradient(w)
        # stopping criterion part 2: if norm of gradient is small, stop
        if scipy.linalg.norm(g) < 1e-4:
            break
        # perform one gradient descent step
        w -= eta*g

    # return the optimized weight
    return w, j+1

# open pdf file
pdf = PdfPages("Surface.pdf")

# start 10 trials with different initial weights
for trials in range(10):
    # create random weights in range [-10, 10]
    w = numpy.random.random(2) * 20 - 10
    # perform gradient descent (copy w to keep original value)
    o, iterations = gradient_descent(w.copy(), 0.04)

    # plot surface
    fig, ax = plot_surface(.5)
    # compute z-values for initial and optimal weights
    loss_w = compute_loss(w[0], w[1])
    loss_o = compute_loss(o[0], o[1])
    # plot values, connected with a line
    ax.plot([w[0], o[0]], [w[1], o[1]], [loss_w, loss_o], "kx-")
    pdf.savefig(fig)

    # print the number of iterations, the start and the final
    print(iterations, w, o, loss_o)

# finalize and close pdf file
pdf.close()

# show plots
pyplot.show()