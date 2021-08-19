import numpy
import random
# # Enable LaTeX interpreter for strings in matplotlib
import matplotlib
# matplotlib.rcParams["text.usetex"] = True
from matplotlib import pyplot

# create training data; assure x_0 = 1 in any case
# parameters are: mean per dimension, std per dimension, size
neg = numpy.random.normal((1,-5,3), (0,2,2), (100,3))
pos = numpy.random.normal((1,3,-5), (0,2,2), (100,3))

# collect in one list
samples = [(sample, -1) for sample in neg] + [(sample, 1) for sample in pos]

# initialize weights
w = numpy.random.normal(size=3)

# stopping criterion: iterate until all samples are classified correctly
# count number of incorrectly classified samples
incorrect = len(samples)
while incorrect > 0:
    # randomly shuffle list
    random.shuffle(samples)
    # iterate over all samples
    incorrect = 0
    for x,t in samples:
        # predict class
        if numpy.dot(w,x) * t < 0:
            incorrect += 1
            w += t*x

# create figure in square shape
pyplot.figure(figsize=(6,6))

# plot points
pyplot.plot(neg[:,1], neg[:,2], "rx")
pyplot.plot(pos[:,1], pos[:,2], "gx")

# compute intersection from plane with z=0
# w_0 + w_1x_1 + w_2x_2 = 0
# => x_2 = (- w_0 - w_1x_1) / w_2
x_1 = numpy.array((-10.,10.))
x_2 = (-w[0] - w[1] * x_1) / w[2]

# plot line
pyplot.plot(x_1,x_2,"b-")

# finalize plot
pyplot.xlim((-10,10))
pyplot.ylim((-10,10))
# pyplot.xlabel("$x_1$")
# pyplot.ylabel("$x_2$")
pyplot.xlabel("x_1")
pyplot.ylabel("x_2")

# write to file
pyplot.savefig("Perceptron.pdf")

# show figure and reset
pyplot.show()