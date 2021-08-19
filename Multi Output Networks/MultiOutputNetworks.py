import numpy
import scipy.linalg
from matplotlib import pyplot

def shuffle_Data(X, T):
    D_1 = X.shape[0]
    Data = numpy.vstack((X, T))
    numpy.random.shuffle(Data.T)
    return Data[:D_1], Data[D_1:]

def batch(X, T, B):
    X, T = shuffle_Data(X, T)
    N = X.shape[1]
    i = 0
    index = 0
    while i < e:
        if index + B >= N:
            i += 1
            first_part_batch_X = X[:, index:N]
            first_part_batch_T = T[:, index:N]
            X, T = shuffle_Data(X, T)
            second_part_batch_X = X[:, :index+B-N]
            second_part_batch_T = T[:, :index + B - N]
            yield numpy.hstack((first_part_batch_X, second_part_batch_X)), numpy.hstack((first_part_batch_T, second_part_batch_T)), True
            index = index + B - N
        else:
            yield X[:, index:index+B], T[:, index:index+B], False
            index = index + B

def logistic(X):
    return 1./(1+numpy.exp(-X))

def forward(X, W1, W2):
    A = numpy.dot(W1,X)
    H = logistic(A)
    H[0] = 1.
    return numpy.dot(W2, H), H

def descent(X, T, W1, W2, eta):
    # network output and hidden states for all inputs
    Y, H = forward(X, W1, W2)
    N = X.shape[1]
    # gradient for W2
    G2 = 2/N * numpy.dot((Y-T), H.T)
    # gradient for W1
    G1 = 2/N * numpy.dot(
        numpy.dot(W2.T, Y-T) * H * (1-H),
        X.T
    )
    # update weights
    W1 -= eta * G1
    W2 -= eta * G2

    Y,_ = forward(X, W1, W2)
    J = scipy.linalg.norm(Y - T, "fro") ** 2 / N

    return W1, W2, J

my_data = numpy.empty(33)
# with open('.\Ex4\student-mat.csv', 'rt') as file:
with open('student-mat.csv', 'rt') as file:
    for line in file:
        x = numpy.array(line.rstrip().split(';'))
        my_data = numpy.vstack((my_data, x))
my_data = numpy.delete(my_data,[0,1],axis=0)
my_data = numpy.hstack((my_data[:,:8], my_data[:,12:]))
my_data = numpy.delete(my_data,numpy.s_[29:],axis=1)

my_data[:,0] = [1. if x == "GP" else -1. for x in my_data[:,0]]
my_data[:,1] = [1. if x == "M" else -1. for x in my_data[:,1]]
my_data[:,3] = [1. if x == "U" else -1. for x in my_data[:,3]]
my_data[:,4] = [1. if x == "GT3" else -1. for x in my_data[:,4]]
my_data[:,5] = [1. if x == "T" else -1. for x in my_data[:,5]]
my_data[:,11] = [1. if x == "yes" else -1. for x in my_data[:,11]]
my_data[:,12] = [1. if x == "yes" else -1. for x in my_data[:,12]]
my_data[:,13] = [1. if x == "yes" else -1. for x in my_data[:,13]]
my_data[:,14] = [1. if x == "yes" else -1. for x in my_data[:,14]]
my_data[:,15] = [1. if x == "yes" else -1. for x in my_data[:,15]]
my_data[:,16] = [1. if x == "yes" else -1. for x in my_data[:,16]]
my_data[:,17] = [1. if x == "yes" else -1. for x in my_data[:,17]]
my_data[:,18] = [1. if x == "yes" else -1. for x in my_data[:,18]]
my_data = my_data.astype(numpy.float)

X = my_data[:, :26].T
T = my_data[:, 26:].T

# add x_0 to the input
X = numpy.vstack((numpy.ones((1,X.shape[1])), X))

# Parameters of 2-layer network
K = 10
eta = 0.005
e = 10000
B = 32

# randomly initialize weights
W1_initial = numpy.random.random((K+1,X.shape[0])) * 2. - 1.
W2_initial = numpy.random.random((3,K+1)) * 2. - 1.

W1 = W1_initial.copy()
W2 = W2_initial.copy()
progression_SGD = []
for X_batch, T_batch, epoch_flag in batch(X, T, B):
    W1, W2, J = descent(X_batch, T_batch, W1, W2, eta)
    if epoch_flag:
        progression_SGD.append(J)
        print(len(progression_SGD))

W1 = W1_initial.copy()
W2 = W2_initial.copy()
progression_GD = []
for X_batch, T_batch, epoch_flag in batch(X, T, X.shape[1]):
    W1, W2, J = descent(X_batch, T_batch, W1, W2, eta)
    if epoch_flag:
        progression_GD.append(J)
        print(len(progression_GD))

# plot loss progression
pyplot.figure()
pyplot.plot(progression_SGD, label="Stochastic Gradient Descent")
pyplot.plot(progression_GD, label="Gradient Descent")
pyplot.legend()
pyplot.xlabel("Epochs")
pyplot.xscale("log")
pyplot.ylabel("Loss")
pyplot.yscale("log")
pyplot.savefig("Loss.pdf")
pyplot.show()