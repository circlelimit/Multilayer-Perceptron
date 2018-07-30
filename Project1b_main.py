exec(open('project_1b.py').read())
import time

print('loading training and testing data...')

trainX, trainY, testX, testY = load_data()

print('Splitting into 5 folds...')
K_folds_X, K_folds_Y = split_K_sets(trainX, trainY)

print('3-layer Multineural network with 5-fold cross validatoon...')

epochs = 1000
batch_size = 32
no_hidden1 = 30 #num of neurons in hidden layer 1
learning_rate = 0.001
num_layer = 3

K_folds_cost = [] #K_folds_cost is initiated to record the best cost in each fold
K_folds_accuracy = [] #K_folds_accuracy is initiated to record the best accuracy in each fold

for fold_num in [x for x in K_folds_Y]:
    
    print (fold_num + '...')
    
    #select 1 fold as validation and the other 4 folds as training
    trainX, trainY, validationX, validationY = split_train_validation(K_folds_X, K_folds_Y, fold_num)
    
    #define & train model based on the training data
    best_cost, best_accuracy = define_model2(num_layer, trainX, trainY, validationX, validationY, epochs, batch_size, no_hidden1, learning_rate)    
    
    K_folds_cost = np.append(K_folds_cost, best_cost)
    K_folds_accuracy = np.append(K_folds_accuracy, best_accuracy)
    
print ('Average cost: %.3e'% (sum(K_folds_cost)/5))
print ('Average accuracy : %1f' % (sum(abs(K_folds_accuracy))/5))

print(time.ctime())

print('Finding the optimal learning rate...')

epochs = 1000
batch_size = 32
no_hidden1 = 30 #num of neurons in hidden layer 1
learning_rate_range = [1e-5,0.5*1e-4, 1e-4, 0.5*1e-3, 1e-3]
num_layer = 3


for learning_rate in learning_rate_range:
    
    K_folds_cost = [] #K_folds_cost is initiated to record the best cost in each fold
    K_folds_accuracy= []
    
    for fold_num in [x for x in K_folds_Y]:
        print (fold_num + '...') 
        
        #select 1 fold as validation and the other 4 folds as training
        trainX, trainY, validationX, validationY = split_train_validation(K_folds_X, K_folds_Y, fold_num)
        
        #define & train model based on the training data
        best_cost, best_accuracy = define_model2(num_layer, trainX, trainY, validationX, validationY, epochs, batch_size, no_hidden1, learning_rate)          
        
        K_folds_cost = np.append(K_folds_cost, best_cost)
        K_folds_accuracy = np.append(K_folds_accuracy, best_accuracy)
        
    average_cost = sum(K_folds_cost)/5
    average_accuracy = sum(abs(K_folds_accuracy))/5
    
    validation_average_cost = np.append(validation_average_cost, average_cost)
    validation_average_accuracy = np.append(validation_average_accuracy, average_accuracy)
    
    print ('Learning rate %.3e, Average cost: %.3e, Average accuracy: %.1f'% (learning_rate, average_cost, average_accuracy))
    
    
print(time.ctime())

print('plotting average cross validation cost aginst different learning rate...')

plot_learning_rate(validation_average_cost, validation_average_accuracy, learning_rate_range)

print (validation_average_cost)
print (validation_average_accuracy)
print (learning_rate_range)

print(time.ctime())

print('Applying the model with the optimal learning rate...')

optimal_learning_rate = 5e-5

trainX, trainY, testX, testY = load_data()
best_cost = define_model2(num_layer, trainX, trainY, testX, testY, epochs, batch_size, no_hidden1, optimal_learning_rate) 

print(time.ctime())

print('Finding the optimal number of hidden neurons...')

epochs = 1000
batch_size = 32
learning_rate = 1e-4 #optimal learning rate found above
no_hidden1_range = [20,30,40,50,60]
num_layer = 3

#Training the model based on the 70% train data with 5-fold cross validation

validation_average_cost = []
validation_average_accuracy = []

for no_hidden1 in no_hidden1_range:
    
    K_folds_cost = [] #K_folds_cost is initiated to record the best cost in each fold
    K_folds_accuracy= []
    
    for fold_num in [x for x in K_folds_Y]:
        print (fold_num + '...') 
        
        #select 1 fold as validation and the other 4 folds as training
        trainX, trainY, validationX, validationY = split_train_validation(K_folds_X, K_folds_Y, fold_num)
        
        #define & train model based on the training data
        best_cost, best_accuracy = define_model2(num_layer, trainX, trainY, validationX, validationY, epochs, batch_size, no_hidden1, learning_rate)          
        
        K_folds_cost = np.append(K_folds_cost, best_cost)
        K_folds_accuracy = np.append(K_folds_accuracy, best_accuracy)
        
    average_cost = sum(K_folds_cost)/5
    average_accuracy = sum(K_folds_accuracy)/5
    
    validation_average_cost = np.append(validation_average_cost, average_cost)
    validation_average_accuracy = np.append(validation_average_accuracy, average_accuracy)
    
    print ('Learning rate %.3e, Average cost: %.3e, Average accuracy: %.1f'% (learning_rate, average_cost, average_accuracy))

print(time.ctime())

print('Plotting average cross validation cost against different number of hidden neurons...')

plot_hidden_neurons(validation_average_cost, validation_average_accuracy, no_hidden1_range)
      
print (validation_average_cost)
print (validation_average_accuracy)
print (no_hidden1_range)

print(time.ctime())

print('Applying the optimal number of hidden neurons...')

# Apply the model with optimal learning rate & optimal number of hidden neurons on the 30% test set
optimal_learning_rate = 5e-5
60timal_no_hidden1 = 60

trainX, trainY, testX, testY = load_data()
best_cost = define_model2(3, trainX, trainY, testX, testY, epochs, batch_size, optimal_no_hidden1, optimal_learning_rate) 

print(time.ctime())

print('4-layer Multiplayer feedforward Neural Network model...')

epochs = 1000
batch_size = 32
no_hidden1= optimal_no_hidden1
learning_rate = 1e-4  #optimal learning rate found
num_layer = 4

# Apply 4-layer NN model with optimal number of hidden neurons on the 30% test set
trainX, trainY, testX, testY = load_data()
train, test = define_model2(4, trainX, trainY, testX, testY, epochs, batch_size, optimal_no_hidden1, learning_rate)


# Apply 4-layer NN model with optimal number of hidden neurons on the 30% test set
trainX, trainY, testX, testY = load_data()
train, test = define_model2(4, trainX, trainY, testX, testY, epochs, batch_size, optimal_no_hidden1, learning_rate)

print(time.ctime())

print('5-layer Multilayer feedforward neural network model...')

epochs = 1000
batch_size = 32
no_hidden1= optimal_no_hidden1
no_hidden2= 20  #num of neurons in hidden layer 2
learning_rate = 1e-4
num_layer = 5

# Apply 5-layer NN model with optimal number of hidden neurons on the 30% test set

trainX, trainY, testX, testY = load_data()
train, test = define_model2(5, trainX, trainY, testX, testY, epochs, batch_size, optimal_no_hidden1, learning_rate)

print(time.ctime())



