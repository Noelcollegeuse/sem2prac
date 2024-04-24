#SLP
def pred( row, weights):
    activaiton =  weights[-1]
    for i in range(len(row)-1):
        activaiton += weights[i+1] * row[i]
    return 1.0 if activaiton >= 0.0 else 0.0

def weight_train(l_rate,n_epoch,train):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            predicition = pred(row, weights)
            error = row[1]- predicition 
            sum_error += error**2
            weights[0]= weights[0] + l_rate * error

            for i in range(len(row)-1):
    
                weights[i+1] = weights[i+1] + l_rate * error * row[i]
        print(epoch,l_rate,error)
    return weights

dataset = [[2.7810836,2.550537003,0], 

[1.465489372,2.362125076,0], 

[3.396561688,4.400293529,0], 

[1.38807019,1.850220317,0], 

[3.06407232,3.005305973,0], 

[7.627531214,2.759262235,1], 

[5.332441248,2.088626775,1], 

[6.922596716,1.77106367,1], 

[8.675418651,-0.242068655,1], 

[7.673756466,3.508563011,1]] 

l_rate = 0.5 

epoch = 5

weights =  weight_train(l_rate,epoch,dataset)

print(weights)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#MLP
from math import exp
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(inputs)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#RBF
import math
import pandas as pd
import numpy as numpy
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
Data= pd.read_table("bank_full.csv", sep= None, engine= "python")
cols= ["age","balance","day","duration","campaign","pdays","previous"]
data_encode= Data.drop(cols, axis= 1)
data_encode= data_encode.apply(LabelEncoder().fit_transform)
data_rest= Data[cols]
Data= pd.concat([data_rest,data_encode], axis= 1)
 
data_train, data_test= train_test_split(Data, test_size= 0.5, random_state= 4)
data_test = data_test[:-1]
X_train= data_train.drop("y", axis= 1)
Y_train= data_train["y"]
X_test= data_test.drop("y", axis=1)
Y_test= data_test["y"]
 
print("Printing Data values: \n")
print(Data.shape)
print(data_train.shape)
print(data_test.shape)
 
scaler= StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
 
K_cent = 8 #number of centers
kmean = KMeans(n_clusters= K_cent,  max_iter= 100, n_init=10)
kmean.fit(X_train)
cent = kmean.cluster_centers_
 
max = 0
for i in range(K_cent):
  for j in range(K_cent):
    d= numpy.linalg.norm(cent[i]-cent[j])
    if(d > max):
      max = d
d = max
sigma = d/math.sqrt(2*K_cent)
 
# G is the Gaussian Function matrix
row = X_train.shape[0]
column = K_cent
G_train = numpy.empty((row,column), dtype= float)
for i in range(row):
 for j in range(column):
  dist= numpy.linalg.norm(X_train[i]-cent[j])
  G_train[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
 
print("\nGaussian matrix (G): \n",G_train)
 
G_test= numpy.empty((row,column), dtype= float)
for i in range(row):
  for j in range(column):
    dist= numpy.linalg.norm(X_test[i]-cent[j])
    G_test[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
 
# W is the Weight matrix
# W = G_inv * x
GTG = numpy.dot(G_train.T,G_train)
GTG_inv = numpy.linalg.inv(GTG)
res = numpy.dot(GTG_inv,G_train.T)
W = numpy.dot(res,Y_train)
 
print("\nWeight Matrix (W): \n", W)
 
predict = numpy.dot(G_test,W)
predict = 0.5*(numpy.sign(predict-0.5)+1)
#print("\nPredict values:\n",predict)
score= accuracy_score(predict, Y_test)
print("\nAccuracy: ",score.mean())
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#CAM
import numpy as np
import random
mem_vectors = [
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0]
]
q = len(mem_vectors) # number of vectors = 3
n = len(mem_vectors[0]) # dimensionality of vectors = 6
bip_mem_vecs = 2*np.array(mem_vectors) - 1
#Initialize and compute the weight matrix
zd_wt_mat = np.zeros((n,n))
for i in range(q):
  zd_wt_mat += np.outer(bip_mem_vecs[i, :], bip_mem_vecs[i, :])
 
zd_wt_mat -= q * np.eye(n) #Zero diagonal
probe = np.array([1, 0, 0, 0, 1, 1])
print(f'The input vector is: {probe}')
signal_vector = 2*probe-1
flag = 0 #Initialize flag
 
while flag != n:
    permindex = np.random.permutation(n)  # Randomize order
    old_signal_vector = np.copy(signal_vector)
    # Update all neurons once per epoch
    for j in range(n):
        act_vec = np.dot(signal_vector, zd_wt_mat)
        if act_vec[permindex[j]] > 0:
            signal_vector[permindex[j]] = 1
        elif act_vec[permindex[j]] < 0:
            signal_vector[permindex[j]] = -1
    flag = np.dot(signal_vector, old_signal_vector)
 
print(f'The recalled vector is: {0.5 * (signal_vector + 1)}')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#BAM
import numpy as np
 
n = 5  # Dimension of Fx
p = 4  # Dimension of Fy
q = 2  # Number of associations
 
# Specify Fx and Fy vectors
mem_vectorsX = np.array([[0, 1, 0, 1, 0], [1, 1, 0, 0, 0]])
mem_vectorsY = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
 
bip_mem_vecsX = 2 * mem_vectorsX - 1  # Convert to bipolar
bip_mem_vecsY = 2 * mem_vectorsY - 1
wt_matrix = np.zeros((n, p))  # Initialize weight matrix
for i in range(q):  # and recursively compute
    wt_matrix += np.outer(bip_mem_vecsX[i], bip_mem_vecsY[i])
 
k = 1  # Set up time index
probe = np.array([0, 1, 0, 1, 1])  # Set up probe
signal_x = 2 * probe - 1  # Set Fx signals to probe
signal_y = np.random.randint(2, size=p)
pattern_x = np.empty((q, n), dtype=int)
pattern_y = np.empty((q, p), dtype=int)
pattern_x[k, :] = signal_x
pattern_y[k, :] = signal_y
 
flag = 0
while flag != 1:
    act_y = np.dot(signal_x, wt_matrix)  # Compute Fx activations
    signal_y = np.where(act_y > 0, 1, -1)  # Set up signals
    if k > 1:
        compare_y = np.array_equal(signal_y, pattern_y[k - 1])
    else:
        compare_y = False
    pattern_y[k] = signal_y  # Store the signal on Fy
    act_x = np.dot(signal_y, wt_matrix.T)  # Compute activations of Fx
    signal_x = np.where(act_x > 0, 1, -1)  # Set up signals
    compare_x = np.array_equal(signal_x, pattern_x[k - 1]) if k > 1 else False
    pattern_x[k] = signal_x  # and store the signal on Fx
    k += 1  # Increment time
    if k >= q:  # Check if k exceeds the size of pattern_x and pattern_y
        break
    flag = compare_x * compare_y  # Check for bidirectional eqlm.
 
# Display update traces
print(f"Probe: {probe}")
print(f"\npattern_x: {pattern_x}")
print(f"\npattern_y: {pattern_y}")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Fuzzy Logic
import numpy as np 

class myFuzzySet: 

    def __init__(self, a, b, c): 

        self.a = a  # Left 

        self.b = b  # Center 

        self.c = c  # Right 

  

    def membership(self, x): 

        if x <= self.b: 

            return (x - self.a) / (self.b - self.a) 

        elif x <= self.c: 

            return (self.c - x) / (self.c - self.b) 

        else: 

            return 0 

    def area(self): 

        return (self.c - self.b) 

  

temperature_below_average = myFuzzySet(15, 30, 45) 

temperature_low = myFuzzySet(-5, 10, 25) 

pressure_below_average = myFuzzySet(1.25, 2 ,2.75) 

pressure_low = myFuzzySet(0.25 , 1, 1.75) 

heating_power_medium_high = myFuzzySet(3.25, 4 ,4.75) 

heating_power_high = myFuzzySet(4.25, 5, 5.75) 

valve_opening_medium_low = myFuzzySet(1.25, 2 , 2.75) 

valve_opening_small = myFuzzySet(0.25,  1, 1.75) 

  

temperature_input = 16.5 

pressure_input = 1.3 

z1 = min(temperature_below_average.membership(temperature_input), pressure_below_average.membership(pressure_input)) 

z2 = min(temperature_low.membership(temperature_input), pressure_low.membership(pressure_input)) 

print("z1 =", z1) 

print("z2 =", z2) 

  

c1num = (z1*heating_power_medium_high.area()*heating_power_medium_high.b)+(z2*heating_power_high.area()*heating_power_high.b) 

c2num = (z1*valve_opening_medium_low.area()*valve_opening_medium_low.b)+(z2*valve_opening_small.area()*valve_opening_small.b) 

c1den = (z1*heating_power_medium_high.area()+z2*heating_power_high.area()) 

c2den = (z1*valve_opening_medium_low.area()+z2*valve_opening_small.area()) 

c1 = c1num/c1den 

c2 = c2num/c2den 

print("C1 and C2 =", c1,c2) 
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
