import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

floatX = theano.config.floatX


# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def load_data():
     #read and divide data into test and train sets
 	cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
 	X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
 	Y_data = (np.asmatrix(Y_data)).transpose()

 	X_data, Y_data = shuffle_data(X_data, Y_data)

 	X_data_mean, X_data_std = np.mean(X_data, axis=0), np.std(X_data, axis=0)

 	X_data = normalize(X_data, X_data_mean, X_data_std)

 	#separate train and test data
 	m = 3*X_data.shape[0] // 10
 	testX, testY = X_data[:m],Y_data[:m]
 	trainX, trainY = X_data[m:], Y_data[m:]

 	"""
 	# scale and normalize data
 	trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
 	testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

 	trainX = scale(trainX, trainX_min, trainX_max)
 	testX = scale(testX, testX_min, testX_max)

 	trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
 	testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

 	trainX = normalize(trainX, trainX_mean, trainX_std)
 	testX = normalize(testX, testX_mean, testX_std)
 	"""

 	return trainX, trainY, testX, testY


def split_K_sets(trainX, trainY):
	"""
	This function splits the training data
	into 5 sets with equal size / size differing by 1
	OUTPUT:
	K_folds_X: a dictionary that contains an array of input training data with name 'Fold X'
	where X = 1, 2, 3, 4, 5
	K_folds_Y: a dictionary that contains an array of output training data with name 'Fold X'
	where X = 1, 2, 3, 4, 5
	"""
	print(trainX.shape)
	rem = trainX.shape[0]%5
	size = int((trainX.shape[0]-rem)/5)
	index = [size*i for i in range(6)]
	plus = [0, 1, 2, 3, 3, 3]
	index = [index[i]+plus[i] for i in range(len(plus))]

	K_folds_X = {}
	K_folds_Y = {}
	for i in range(len(index)-1):
	    K_folds_X['Fold ' + str(i+1)] = trainX[index[i]:index[i+1]]
	    K_folds_Y['Fold ' + str(i+1)] = trainY[index[i]:index[i+1]]
	    print (K_folds_X['Fold ' + str(i+1)].shape)
	return K_folds_X, K_folds_Y

def split_train_validation (K_folds_X,K_folds_Y,fold_num):
	#fold_num: the fold number that is selected to be the VALIDATION set.
	#the other 4 folds will be used for training the model
	validationX, validationY = K_folds_X[fold_num], K_folds_Y[fold_num]
	folds = [x for x in K_folds_Y]
	folds.remove(fold_num)
	trainX = np.concatenate([K_folds_X[K] for K in folds])
	trainY = np.concatenate([K_folds_Y[K] for K in folds])

	return trainX, trainY, validationX, validationY

"""
def define_model(trainX, epochs, batch_size, no_hidden1, learning_rate):

	no_features = trainX.shape[1]
	x = T.matrix('x') # data sample
	d = T.matrix('d') # desired output
	no_samples = T.scalar('no_samples')

	# initialize weights and biases for hidden layer(s) and output layer
	w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX )
	b_o = theano.shared(np.random.randn()*.01, floatX)
	w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
	b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

	# learning rate
	alpha = theano.shared(learning_rate, floatX)


	#Define mathematical expression:
	h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
	y = T.dot(h1_out, w_o) + b_o

	cost = T.abs_(T.mean(T.sqr(d - y)))
	accuracy = T.mean(d - y)

	#define gradients
	dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

	train = theano.function(
	        inputs = [x, d],
	        outputs = cost,
	        updates = [[w_o, w_o - alpha*dw_o],
	                   [b_o, b_o - alpha*db_o],
	                   [w_h1, w_h1 - alpha*dw_h],
	                   [b_h1, b_h1 - alpha*db_h]],
	        allow_input_downcast=True
	        )
	test = theano.function(
	    inputs = [x, d],
	    outputs = [y, cost, accuracy],
	    allow_input_downcast=True
	    )


	train_cost = np.zeros(epochs)
	test_cost = np.zeros(epochs)
	test_accuracy = np.zeros(epochs)


	min_error = 1e+15
	best_iter = 0
	best_w_o = np.zeros(no_hidden1)
	best_w_h1 = np.zeros([no_features, no_hidden1])
	best_b_o = 0
	best_b_h1 = np.zeros(no_hidden1)

	alpha.set_value(learning_rate)
	print('learning rate: %.3e' % alpha.get_value())
	return train, test
"""

def plot_training_validation(train_cost, validation_cost):
	plt.figure()
	plt.plot(range(epochs), train_cost, label='train error')
	plt.plot(range(epochs), validation_cost, label = 'validation error')
	plt.xlabel('Time (s)')
	plt.ylabel('Mean Squared Error')
	plt.title('Training and validation Errors at Alpha = %.3f'%learning_rate)
	plt.legend()
	#plt.savefig('p_1b_sample_mse.png')
	plt.show()

def plot_test_accuracy(test_accuracy):
	plt.figure()
	plt.plot(range(epochs), test_accuracy)
	plt.xlabel('Epochs')
	plt.ylabel('MSE')
	plt.title('Test MSE')
	#plt.savefig('p_1b_sample_accuracy.png')
	plt.show()


def plot_learning_rate(validation_average_cost, validation_average_accuracy, learning_rate_range):
	plt.figure()
	plt.plot(learning_rate_range, validation_average_cost)
	plt.xlabel('Learning rate')
	plt.ylabel('5-fold validation cost')
	plt.title('CV validation cost with varying learning rates')
	#plt.savefig('p_1b_sample_accuracy.png')

	plt.figure()
	plt.plot(learning_rate_range, validation_average_accuracy)
	plt.xlabel('Learning rate')
	plt.ylabel('5-fold validation cost')
	plt.title('CV validation accuracy with varying learning rates')

	plt.show()


def plot_hidden_neurons(validation_average_cost,validation_average_accuracy, no_hidden1_range):
	plt.figure()
	plt.plot(no_hidden1_range, validation_average_cost)
	plt.xlabel('Number of hidden neurons')
	plt.ylabel('5-fold validation cost')
	plt.title('CV validation cost with varying number of hidden neurons')
	#plt.savefig('p_1b_sample_accuracy.png')

	plt.figure()
	plt.plot(no_hidden1_range, validation_average_accuracy)
	plt.xlabel('Number of hidden neurons')
	plt.ylabel('5-fold validation accuracy')
	plt.title('CV validation accuracy with varying number of hidden neurons')

	plt.show()


def define_model2(num_layer,trainX, trainY, validationX, validationY, epochs, batch_size, no_hidden1, learning_rate):
	"""
	Thif function defines NN model.
	INPUT:
	num_layer: the number of layers in the NN model (3, 4, 5)
	trainX : the data set to be trained
	epochs: the number of iterations
	batch_size: the size of each mini-batch
	no_hidden1: the number of hidden neurons for the 1st hidden layer
	(the numer of hidden neurons in 2nd and 3rd layers is default: 20)
	learning_rate = learning rate of the model, alpha
	"""

	no_features = trainX.shape[1]
	x = T.matrix('x') # data sample
	d = T.matrix('d') # desired output
	no_samples = T.scalar('no_samples')

	# initialize weights and biases for hidden layer(s) and output layer
	w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
	b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

	if num_layer == 3:
		w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX )
		b_o = theano.shared(np.random.randn()*.01, floatX)
	if num_layer == 4:
		print('ok')
		w_h2 = theano.shared(np.random.randn(no_hidden1, 20)*.01, floatX )
		b_h2 = theano.shared(np.random.randn(20)*0.01, floatX)
		w_o = theano.shared(np.random.randn(20)*.01, floatX )
		b_o = theano.shared(np.random.randn()*.01, floatX)
	if num_layer == 5:
		print ('ok')
		w_h2 = theano.shared(np.random.randn(no_hidden1, 20)*.01, floatX )
		b_h2 = theano.shared(np.random.randn(20)*0.01, floatX)
		w_h3 = theano.shared(np.random.randn(20, 20)*.01, floatX )
		b_h3 = theano.shared(np.random.randn(20)*0.01, floatX)
		w_o = theano.shared(np.random.randn(20)*.01, floatX )
		b_o = theano.shared(np.random.randn()*.01, floatX)


	# learning rate
	alpha = theano.shared(learning_rate, floatX)

	#Define mathematical expression:
	h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
	if num_layer == 3:
		y = T.dot(h1_out, w_o) + b_o
		cost = T.abs_(T.mean(T.sqr(d - y)))
		accuracy = T.mean(T.sqr(d - y))
		#define gradients
		dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

	if num_layer == 4:
		print('ok')
		h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
		y = T.dot(h2_out, w_o) + b_o

		cost = T.abs_(T.mean(T.sqr(d - y)))
		accuracy = T.mean(T.sqr(d - y))
		#define gradients
		dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2])

	if num_layer == 5:
		print ('ok')
		h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
		h3_out = T.nnet.sigmoid(T.dot(h2_out, w_h3) + b_h3)
		y = T.dot(h3_out, w_o) + b_o
		cost = T.abs_(T.mean(T.sqr(d - y)))
		accuracy = T.mean(T.sqr(d - y))
		#define gradients
		dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2, dw_h3, db_h3 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2, w_h3, b_h3])


	if num_layer == 3:
		train = theano.function(
		        inputs = [x, d],
		        outputs = cost,
		        updates = [[w_o, w_o - alpha*dw_o],
		                   [b_o, b_o - alpha*db_o],
		                   [w_h1, w_h1 - alpha*dw_h],
		                   [b_h1, b_h1 - alpha*db_h]],
		        allow_input_downcast=True
		        )

	if num_layer == 4:
		print ('ok')
		train = theano.function(
		        inputs = [x, d],
		        outputs = cost,
		        updates = [[w_o, w_o - alpha*dw_o],
		                   [b_o, b_o - alpha*db_o],
		                   [w_h1, w_h1 - alpha*dw_h1],
		                   [b_h1, b_h1 - alpha*db_h1],
		                   [w_h2, w_h2 - alpha*dw_h2],
		                   [b_h2, b_h2 - alpha*db_h2]],
		        allow_input_downcast=True
		        )

	if num_layer == 5:
		print ('ok')
		train = theano.function(
		        inputs = [x, d],
		        outputs = cost,
		        updates = [[w_o, w_o - alpha*dw_o],
		                   [b_o, b_o - alpha*db_o],
		                   [w_h1, w_h1 - alpha*dw_h1],
		                   [b_h1, b_h1 - alpha*db_h1],
		                   [w_h2, w_h2 - alpha*dw_h2],
		                   [b_h2, b_h2 - alpha*db_h2],
		                   [w_h3, w_h3 - alpha*dw_h3],
		                   [b_h3, b_h3 - alpha*db_h3]],
		        allow_input_downcast=True
		        )

	test = theano.function(
		   inputs = [x, d],
		   outputs = [y, cost, accuracy],
		   allow_input_downcast=True
		   )


	train_cost = np.zeros(epochs)
	test_cost = np.zeros(epochs)
	test_accuracy = np.zeros(epochs)


	min_error = 1e+15
	best_iter = 0
	best_b_o = 0
	best_w_h1 = np.zeros([no_features, no_hidden1])
	best_b_h1 = np.zeros(no_hidden1)

	if num_layer == 3:
		best_w_o = np.zeros(no_hidden1)

	if num_layer == 4:
		print ('ok')
		best_w_h2 = np.zeros([no_hidden1, 20])
		best_b_h2 = np.zeros(20)
		best_w_o = np.zeros(20)

	if num_layer ==5:
		print ('ok')
		best_w_h2 = np.zeros([no_hidden1, 20])
		best_b_h2 = np.zeros(20)
		best_w_h3 = np.zeros([20, 20])
		best_b_h3 = np.zeros(20)
		best_w_o = np.zeros(20)

	alpha.set_value(learning_rate)
	print('learning rate: %.3e' % alpha.get_value())
	print ('number of hidden neurons in layer 1: %d' % no_hidden1)

	#start training model
	for iter in range(epochs):
	    #if iter % 100 == 0:
	        #print(iter)

	    trainX, trainY = shuffle_data(trainX, trainY)
	    train_cost[iter] = train(trainX, np.transpose(trainY))
	    pred, test_cost[iter], test_accuracy[iter] = test(validationX, np.transpose(validationY))

	    if test_cost[iter] < min_error:
	        best_iter = iter
	        min_error = test_cost[iter]
	        best_w_o = w_o.get_value()
	        best_w_h1 = w_h1.get_value()
	        best_b_o = b_o.get_value()
	        best_b_h1 = b_h1.get_value()

	        if num_layer == 4:
	        	best_w_h2 = w_h2.get_value()
	        	best_b_h2 = b_h2.get_value()

	        if num_layer ==5:
	        	best_w_h2 = w_h2.get_value()
	        	best_b_h2 = b_h2.get_value()
	        	best_w_h3 = w_h3.get_value()
	        	best_b_h3 = b_h3.get_value()



	plot_training_validation(train_cost, test_cost)

	#set weights and biases to values at which performance was best
	w_o.set_value(best_w_o)
	b_o.set_value(best_b_o)
	w_h1.set_value(best_w_h1)
	b_h1.set_value(best_b_h1)

	if num_layer == 4:
		w_h2.set_value(best_w_h2)
		b_h2.set_value(best_b_h2)
	if num_layer == 5:
		w_h2.set_value(best_w_h2)
		b_h2.set_value(best_b_h2)
		w_h3.set_value(best_w_h3)
		b_h3.set_value(best_b_h3)

	best_pred, best_cost, best_accuracy = test(validationX, np.transpose(validationY))

	plot_test_accuracy(test_accuracy)

	print('Minimum error: %.3e, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))

	return best_cost, best_accuracy
