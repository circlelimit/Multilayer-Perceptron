exec(open('project_1a.py').read())
import time

print ('loading training and testing data...')

trainX, trainY, testX, testY = load_data()

print(time.ctime())

print('Finding optimal batch size...')

decay = 1e-6
learning_rate = 0.01
epochs = 1000
num_neurons_L1 = 10
batch_size_range = [4, 8, 16, 32, 64]

min_train_cost = []
max_test_accuracy = []
time_update = []
train_cost_dic = {}
test_accuracy_dic = {}

random.seed(datetime.now())
train, predict = define_model(decay, learning_rate, epochs, num_neurons_L1)

for batch_size in batch_size_range:
    print('Batch size: %d'% (batch_size))
    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    
    start = timeit.default_timer()
    
    for i in range(epochs):
        if i % 1000 == 0:
            print(i)
            
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        
        train_cost = np.append(train_cost, cost/(n // batch_size))
        
        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
    
    stop = timeit.default_timer() 
    time_update = np.append(time_update, (start-stop))
    
    #plot test accuracy and training cost against iterations
    #plot_iterations(epochs, train_cost, test_accuracy)
     
    train_cost_dic['Batch size: '+ str(batch_size)] = train_cost
    test_accuracy_dic['Batch size: '+ str(batch_size)] = test_accuracy
    
    #record minnimum training cost and maximum testing accuracy for each batch size
    min_train_cost = np.append(min_train_cost, np.min(train_cost))
    max_test_accuracy = np.append(max_test_accuracy, np.max(test_accuracy)*100)
    
print(time.ctime())
    
print ('Plotting traning cost and test accuracy against number of iterations...')
plt.figure()
for batch_size in batch_size_range:
    plt.plot(range(epochs), train_cost_dic['Batch size: '+ str(batch_size)], label = 'Batch size' +str(batch_size))
    
plt.legend(loc='best')
plt.xlabel('iterations')
plt.ylabel('cross-entropy')
plt.title('training cost')
#plt.savefig('p1a_sample_cost.png')

plt.figure()
for batch_size in batch_size_range: 
    plt.plot(range(epochs), test_accuracy_dic['Batch size: '+ str(batch_size)], label = 'Batch size' + str(batch_size))
    
plt.legend(loc='best')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy')
#plt.savefig('p1a_sample_accuracy.png')
    
plt.show()

print(time.ctime())

print('plotting training cost and test accuracy against different batch sizes...')
      
plot_batch_size(batch_size_range, min_train_cost, max_test_accuracy)

print(time.ctime())


print('plotting the time taken to update weights against different batch sizes...')

trainX.shape[0]
num_sample = [round(trainX.shape[0])/batch_size for batch_size in batch_size_range]
time_update_each_sample = [time_update[i]/(num_sample[i]*1000) for i in range(len(time_update))]
plot_running_time(batch_size_range, time_update_each_sample)

print(time_update_each_sample)

print(time.ctime())


print('Finding the optimal nuber of hidden neurons...')


decay = 1e-6
learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons_L1_range = [5, 10, 15, 20, 25]


min_train_cost = []
max_test_accuracy = []
time_update = []
train_cost_dic={}
test_accuracy_dic = {}

for num_neurons_L1 in num_neurons_L1_range:
    print('Number of neurons: %d'%(num_neurons_L1))
    train, predict = define_model(decay, learning_rate, epochs, num_neurons_L1)
    
    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    
    start = timeit.default_timer()
    
    for i in range(epochs):
        if i % 1000 == 0:
            print(i)
            
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        
        train_cost = np.append(train_cost, cost/(n // batch_size))
        
        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
    
    train_cost_dic['Hidden neurons number: '+ str(num_neurons_L1)] = train_cost
    test_accuracy_dic['Hidden neurons number: '+ str(num_neurons_L1)] = test_accuracy
    
    stop = timeit.default_timer()
    time_update = np.append(time_update, (start-stop))
    
    #plot test accuracy and training cost against iterations
    #plot_iterations(epochs, train_cost, test_accuracy)
        
    #record minnimum training cost and maximum testing accuracy for each batch size
    min_train_cost = np.append(min_train_cost, np.min(train_cost))
    max_test_accuracy = np.append(max_test_accuracy, np.max(test_accuracy)*100)
    
time.ctime()


print('Plotting training cost and test accuracy against the number of iterations...')

#Plot
plt.figure()
for num_neurons_L1 in num_neurons_L1_range:
    plt.plot(range(epochs), train_cost_dic['Hidden neurons number: '+ str(num_neurons_L1)], label = 'Hidden neurons number: '+ str(num_neurons_L1))
    
plt.legend(loc='best')
plt.xlabel('iterations')
plt.ylabel('cross-entropy')
plt.title('training cost')
#plt.savefig('p1a_sample_cost.png')

plt.figure()
for num_neurons_L1 in num_neurons_L1_range:
    plt.plot(range(epochs), test_accuracy_dic['Hidden neurons number: '+ str(num_neurons_L1)], label = 'Hidden neurons number: '+ str(num_neurons_L1))
    
plt.legend(loc='best')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy')
#plt.savefig('p1a_sample_accuracy.png')
    
plt.show()

time.ctime()

print('Plotting training cost against different number of hidden neurons...')

plot_hidden_neurons(num_neurons_L1_range, min_train_cost, max_test_accuracy)

time.ctime()

print('Plotting time taken to update weights against different number of hidden neurons...')

num_sample =round(trainX.shape[0])/batch_size
time_update_each_sample = [time_update[i]/(num_sample*1000) for i in range(len(time_update))]
plot_running_time(num_neurons_L1_range, time_update_each_sample)


print(time_update_each_sample)


time.ctime()


print('Finding the optimal number of decay...')

learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons_L1 = 10
decay_range = [1e-3, 1e-6, 1e-9, 1e-12, 0]

min_train_cost = []
max_test_accuracy = []
time_update = []
test_accuracy_dic ={}
train_cost_dic ={}

for decay in decay_range:
    print('Decay: %.3e'%(decay))
    
    train, predict = define_model(decay, learning_rate, epochs, num_neurons_L1)
    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    
    start = timeit.default_timer()
    
    for i in range(epochs):
        if i % 1000 == 0:
            print(i)
            
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        
        train_cost = np.append(train_cost, cost/(n // batch_size))
        
        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
    
    train_cost_dic['Decay: '+str(decay)] =train_cost
    test_accuracy_dic['Decay: '+str(decay)] = test_accuracy
    
    stop = timeit.default_timer() 
    time_update = np.append(time_update, (start-stop))
    
    #plot test accuracy and training cost against iterations
    #plot_iterations(epochs, train_cost, test_accuracy)
        
    #record minnimum training cost and maximum testing accuracy for each batch size
    min_train_cost = np.append(min_train_cost, np.min(train_cost))
    max_test_accuracy = np.append(max_test_accuracy, np.max(test_accuracy)*100)

time.ctime()


print('Plotting training cost and test accuracy against the number of iterations...')


#Plot
plt.figure()
for decay in decay_range:
    plt.plot(range(epochs), train_cost_dic['Decay: '+str(decay)], label = 'Decay: '+str(decay))
    
plt.legend(loc='best')
plt.xlabel('iterations')
plt.ylabel('cross-entropy')
plt.title('training cost')
#plt.savefig('p1a_sample_cost.png')

plt.figure()
for decay in decay_range:
    plt.plot(range(epochs), test_accuracy_dic['Decay: '+str(decay)], label = 'Decay: '+str(decay))
    
plt.legend(loc='best')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy')
#plt.savefig('p1a_sample_accuracy.png')
    
plt.show()

time.ctime()

print('4-layer Multilayer Neural Network model...')

learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons_L1 = 10
num_neurons_L2 = 10
decay = 1e-12

min_train_cost = []
max_test_accuracy = []
time_update = []

train_cost = []
test_accuracy = []

train, predict = define_model2(decay, learning_rate, epochs, num_neurons_L1, num_neurons_L2)
    # train and test
    
start = timeit.default_timer()
    
for i in range(epochs):
    if i % 1000 == 0:
        print(i)
            
    trainX, trainY = shuffle_data(trainX, trainY)
    cost = 0.0
        
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
        cost += train(trainX[start:end], trainY[start:end])
    train_cost = np.append(train_cost, cost/(n // batch_size))
    test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
    
stop = timeit.default_timer() 
time_update = np.append(time_update, (start-stop))
    
#plot test accuracy and training cost against iterations
plot_iterations(epochs, train_cost, test_accuracy)
        
#record minnimum training cost and maximum testing accuracy for each batch size
min_train_cost = np.append(min_train_cost, np.min(train_cost))
max_test_accuracy = np.append(max_test_accuracy, np.max(test_accuracy)*100)
    
#print results
#print results
print ('Running time: %f sec'% ((start-stop)/(1000*batch_size)))
print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))

time.ctime()



