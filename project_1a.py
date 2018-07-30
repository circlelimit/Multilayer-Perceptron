import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import timeit
import random
from datetime import datetime

def init_bias(n = 1):
    return(theano.shared(np.zeros(n), theano.config.floatX))

def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(
        np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)),
        dtype=theano.config.floatX
        )
    if logistic == True:
        W_values *= 4
    return (theano.shared(value=W_values, name='W', borrow=True))

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-np.min(X, axis=0))

# update parameters
def sgd(cost, params, lr=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


def define_model(decay, learning_rate, epochs, num_neurons_L1):
    #this function define a 3-layer network.
    #num_neurons_L1 = the number of neurons in this (fist) hidden layer
    
    # theano expressions
    X = T.matrix() #features
    Y = T.matrix() #output
    

    #init_weights(number of in-neurons, number of out neurons)
    #init_bias(number of out neurons)

    w1, b1 = init_weights(36, num_neurons_L1), init_bias(num_neurons_L1) #weights and biases from input to hidden layer
    w2, b2 = init_weights(num_neurons_L1, 6, logistic=False), init_bias(6) #weights and biases from hidden to output layer
    
    h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
    py = T.nnet.softmax(T.dot(h1, w2) + b2)
    
    y_x = T.argmax(py, axis=1)
    
    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1))+T.sum(T.sqr(w2)))
    params = [w1, b1, w2, b2]
    updates = sgd(cost, params, learning_rate)
    
    # compile
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    return train, predict

def load_data():
    #This function read in the training and testing data as the array trainX (input) and trainY (output), testX (input) and testY (output)
    #read train data
    train_input = np.loadtxt('sat_train.txt',delimiter=' ')
    trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
    #trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
    #trainX = scale(trainX, trainX_min, trainX_max)
    
    train_Y[train_Y == 7] = 6
    trainY = np.zeros((train_Y.shape[0], 6))
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1
    
    #read test data
    test_input = np.loadtxt('sat_test.txt',delimiter=' ')
    testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
    
    X_data = np.append(trainX, testX)
    X_min, X_max= np.min(X_data, axis=0), np.max(X_data, axis=0)
    
    #scale training data
    trainX = scale(trainX, X_min, X_max)
    testX = scale(testX, X_min, X_max)
    
    
    #testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
    #testX = scale(testX, testX_min, testX_max)
    
    test_Y[test_Y == 7] = 6
    testY = np.zeros((test_Y.shape[0], 6))
    
    testY[np.arange(test_Y.shape[0]), test_Y-1] = 1
    
    
    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
    
    return trainX, trainY, testX, testY

def train_test_network(trainX,trainY, testX, testY, batch_size):
    #so far this function does not really work 
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    for i in range(epochs):
        if i % 1000 == 0:
            print(i)
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        (train, predict) = define_model(decay, learning_rate, epochs)
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
            
        train_cost = np.append(train_cost, cost/(n // batch_size))
        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
    
    print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))
    return train_cost, test_accuracy

def plot_iterations(epochs, train_cost, test_accuracy):
    #This function plots training cost and test accuracy against number of iterations 
    #epochs = number of loops
    #train_cost = the array of cross-entropy for each iteration
    #test accuracy = the array of test accuracy for each iteration

    #Plots
    plt.figure()
    plt.plot(range(epochs), train_cost)
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    #plt.savefig('p1a_sample_cost.png')

    plt.figure()
    plt.plot(range(epochs), test_accuracy)
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    #plt.savefig('p1a_sample_accuracy.png')
    
    plt.show()

def plot_batch_size(batch_size_range, min_train_cost, max_test_accuracy):
    #Plots
    plt.figure()
    plt.plot(batch_size_range, min_train_cost)
    plt.xlabel('batach size')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    #plt.savefig('p1a_sample_cost.png')

    plt.figure()
    plt.plot(batch_size_range, max_test_accuracy)
    plt.xlabel('batch size')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    #plt.savefig('p1a_sample_accuracy.png')
    
    plt.show()
    
def plot_running_time(factor, running_time):
    #Plots
    plt.figure()
    plt.plot(factor, running_time)
    plt.xlabel('batach size / number of neurons / decay')
    plt.ylabel('Running time')
    plt.title('Time taken to update weights')
    #plt.savefig('p1a_sample_cost.png')
    
    plt.show()

def plot_hidden_neurons(num_neurons_L1, min_train_cost, max_test_accuracy):
    #Plots
    plt.figure()
    plt.plot(num_neurons_L1, min_train_cost)
    plt.xlabel('Number of hidden neurons')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    #plt.savefig('p1a_sample_cost.png')

    plt.figure()
    plt.plot(num_neurons_L1, max_test_accuracy)
    plt.xlabel('Number of hidden neurons')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    #plt.savefig('p1a_sample_accuracy.png')
    
    plt.show()
    

def plot_decay(decay_range, min_train_cost, max_test_accuracy):
    #Plots
    plt.figure()
    plt.plot(decay_range, min_train_cost)
    plt.xlabel('dacay')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    #plt.savefig('p1a_sample_cost.png')

    plt.figure()
    plt.plot(decay_range, max_test_accuracy)
    plt.xlabel('decay')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    #plt.savefig('p1a_sample_accuracy.png')
    
    plt.show()
    

def define_model2(decay, learning_rate, epochs, num_neurons_L1, num_neurons_L2):
    #this function define a 4-layer network.
    #num_neurons_L1 = the number of neurons in the fist hidden layer
    #num_neurons_L2 = the number of neurons in the second hidden layer
    
    # theano expressions
    X = T.matrix() #features
    Y = T.matrix() #output
    

    #init_weights(number of in-neurons, number of out neurons)
    #init_bias(number of out neurons)

    w1, b1 = init_weights(36, num_neurons_L1), init_bias(num_neurons_L1) #weights and biases from input to 1st hidden layer
    w2, b2 = init_weights(num_neurons_L1, num_neurons_L2), init_bias(num_neurons_L2) #weights and biases from input to 2nd hidden layer
    w3, b3 = init_weights(num_neurons_L2, 6, logistic=False), init_bias(6) #weights and biases from hidden to output layer
    
    h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
    h2 = T.nnet.sigmoid(T.dot(h1, w2) + b2)
    py = T.nnet.softmax(T.dot(h2, w3) + b3)
    
    y_x = T.argmax(py, axis=1)
    

    #what is cost here?
    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1))+T.sum(T.sqr(w2))+T.sum(T.sqr(w3)))
    params = [w1, b1, w2, b2, w3, b3]
    updates = sgd(cost, params, learning_rate)
    
    # compile
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    return train, predict