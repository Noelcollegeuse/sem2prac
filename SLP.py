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
