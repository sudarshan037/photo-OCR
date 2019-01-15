import numpy as np
import matplotlib.pyplot as plt

##############################loading dataset##########################
x_train = np.load("dataset/x_train.npy")
y_train = np.load("dataset/y_train.npy")
x_test = np.load("dataset/x_test.npy")
y_test = np.load("dataset/y_test.npy")
print("train set shape", x_train.shape)
print("test set shape", x_test.shape)

#############################checking dataset#########################
index = 30
plt.imshow(x_train[index])
print("y = "+ str(y_train[index]))
##plt.show()

##############################unrolling dataset############################
m_train = x_train.shape[0]
m_test = x_test.shape[0]
numb_pixels = x_train.shape[1]
train_set_x_flatten = x_train.reshape(m_train, -1)
test_set_x_flatten = x_test.reshape(m_test, -1)
##standardize dataset
x_train = train_set_x_flatten/255.
x_test = test_set_x_flatten/255.

########################################################################
##define all the functions()
##########################################################################
##call for training set
##check accuracy
##############################################################################
##call for test set
##check test set accuracy
