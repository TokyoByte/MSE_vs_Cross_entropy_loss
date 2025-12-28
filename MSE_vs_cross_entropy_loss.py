import math

x = 1.0

weight_1 = 0.6
bias_1 = 0.9

weight_2 = 2.0
bias_2 = 2.0

y = 0.0

def loss_MSE(predtcted_value, true_value):
    loss = (predtcted_value - true_value) ** 2
    return loss

def loss_cross_entorpy_loss(predicted_value, true_value):
    loss = -true_value * (math.log(predicted_value)) + (1 - true_value)*math.log((1 - predicted_value))
    return loss

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def backprop_MSE(y_pred, y):
    partial_derivate = (y_pred - y) *  y_pred * (1 - y_pred)
    return partial_derivate

def backprop_cross_entropy_loss(y_pred,y):
    
    partial_derivate = (y_pred - y) # this is wrong a term of x will also come
    return partial_derivate                                 # but we have used x down
                                                            # otherwise we would have to create two functions, one for weight and other for bias
eta = 0.15

epochs = 300

loss = []
output = []
weights = []
biases = []
epochss = []

#weights.append(weight_1)
#biases.append(bias_1)

for epoch in range(epochs):
    
    epochss.append(epoch)

    weights.append(weight_1)
    biases.append(bias_1)

    z = x*weight_1 + bias_1
    y_pred = sigmoid(z)

    output.append(z)

    cost = loss_MSE(y_pred, y)
    loss.append(abs(cost))

    partial_derivate = backprop_MSE(y_pred,y)

    weight_1 = weight_1 - eta * partial_derivate * x

    bias_1 = bias_1 - eta * partial_derivate


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.plot(epochss, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(2, 2, 2)
plt.plot(epochss, output)
plt.xlabel("Epochs")
plt.ylabel("Output")

plt.subplot(2, 2, 3)
plt.plot(epochss, weights)
plt.xlabel("Epochs")
plt.ylabel("Weight")

plt.subplot(2, 2, 4)
plt.plot(epochss, biases)
plt.xlabel("Epochs")
plt.ylabel("Bias")

plt.show()

loss = []
output = []
weights = []
biases = []
epochss = []

print(output)
#weights.append(weight_1)
#biases.append(bias_1)

for epoch in range(epochs):
    
    epochss.append(epoch)

    weights.append(weight_2)
    biases.append(bias_2)

    z = x*weight_2 + bias_2
    y_pred = sigmoid(z)
    output.append(z)

    cost = loss_MSE(y_pred, y)
    loss.append(abs(cost))

    partial_derivate = backprop_MSE(y_pred,y)

    weight_2 = weight_2 - eta * partial_derivate * x

    bias_2 = bias_2 - eta * partial_derivate

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.plot(epochss, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(2, 2, 2)
plt.plot(epochss, output)
plt.xlabel("Epochs")
plt.ylabel("Output")

plt.subplot(2, 2, 3)
plt.plot(epochss, weights)
plt.xlabel("Epochs")
plt.ylabel("Weight")

plt.subplot(2, 2, 4)
plt.plot(epochss, biases)
plt.xlabel("Epochs")
plt.ylabel("Bias")

plt.show()


import math

x = 1.0

weight_1 = 0.6
bias_1 = 0.9

weight_2 = 2.0
bias_2 = 2.0

y = 0.0

def loss_MSE(predtcted_value, true_value):
    loss = (predtcted_value - true_value) ** 2
    return loss

def loss_cross_entorpy_loss(predicted_value, true_value):
    loss = -true_value * (math.log(predicted_value)) + (1 - true_value)*math.log((1 - predicted_value))
    return loss

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def backprop_cross_entropy_loss(y_pred,y):
    
    partial_derivate = (y_pred - y) # this is wrong a term of x will also come
    return partial_derivate                                 # but we have used x down
                                                            # otherwise we would have to create two functions, one for weight and other for bias
eta = 0.15

epochs = 300

loss = []
output = []
weights = []
biases = []
epochss = []

#weights.append(weight_1)
#biases.append(bias_1)

for epoch in range(epochs):
    
    epochss.append(epoch)

    weights.append(weight_1)
    biases.append(bias_1)

    z = x*weight_1 + bias_1
    y_pred = sigmoid(z)

    output.append(z)

    cost = loss_cross_entorpy_loss(y_pred, y)
    loss.append(abs(cost))

    partial_derivate = backprop_cross_entropy_loss(y_pred,y)

    weight_1 = weight_1 - eta * partial_derivate * x

    bias_1 = bias_1 - eta * partial_derivate


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.plot(epochss, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(2, 2, 2)
plt.plot(epochss, output)
plt.xlabel("Epochs")
plt.ylabel("Output")

plt.subplot(2, 2, 3)
plt.plot(epochss, weights)
plt.xlabel("Epochs")
plt.ylabel("Weight")

plt.subplot(2, 2, 4)
plt.plot(epochss, biases)
plt.xlabel("Epochs")
plt.ylabel("Bias")

plt.show()

loss = []
output = []
weights = []
biases = []
epochss = []

print(output)
#weights.append(weight_1)
#biases.append(bias_1)

for epoch in range(epochs):
    
    epochss.append(epoch)

    weights.append(weight_2)
    biases.append(bias_2)

    z = x*weight_2 + bias_2
    y_pred = sigmoid(z)
    output.append(z)

    cost = loss_cross_entorpy_loss(y_pred, y)
    loss.append(abs(cost))

    partial_derivate = backprop_cross_entropy_loss(y_pred,y)

    weight_2 = weight_2 - eta * partial_derivate * x

    bias_2 = bias_2 - eta * partial_derivate

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.plot(epochss, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(2, 2, 2)
plt.plot(epochss, output)
plt.xlabel("Epochs")
plt.ylabel("Output")

plt.subplot(2, 2, 3)
plt.plot(epochss, weights)
plt.xlabel("Epochs")
plt.ylabel("Weight")

plt.subplot(2, 2, 4)
plt.plot(epochss, biases)
plt.xlabel("Epochs")
plt.ylabel("Bias")

plt.show()