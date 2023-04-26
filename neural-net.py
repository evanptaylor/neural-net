import numpy as np
import matplotlib.pyplot as plt

#create custom dataset (spiral) -- function from nnfs.io
def create_spiral_data(points, classes):
    X = np.zeros((points*classes , 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0,1,points) #radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

#graph dataset to see what we're working with (if applicable)
def plot_data(X):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.title("dataset (X)")
    plt.show()

def plot_results(predictions):
    for i in range(len(X)):
        if y[i] == predictions[i]:
            plt.scatter(X[i,0], X[i,1], color='g') #plot green if correct prediction
        else:
            plt.scatter(X[i,0], X[i,1], color='r') #else red
    plt.title("Predictions")
    plt.show()

#Layer object made from n_nuerons nuerons and has n_inputs inputs
class Layer:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.1*np.random.randn(n_inputs, n_nuerons) #initialze weights matrix (n_inputs x n_nuerons) with small random vals
        self.biases = np.zeros((1, n_nuerons)) #initalize biases vector as all zeros
    #forward pass through model
    def forward(self, inputs): #X if first layer else self.output from previous layer
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    #backwards pass for backpropagation
    def backward(self, dvalues):
        #gradient for params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #gradient for inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

#Rectified Linear Unit activation function
    #standard activation function for hidden layers
class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        #gradient is zero where input values were <= 0
        self.dinputs[self.inputs <= 0] = 0

#Softmax -- exponentiation and normalization
    #activtion function for output layer
class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True )) #Overflow prevention -- restrict values to [0,1]
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        #enumerate outputs and gradients
        for index, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            #flatten
            output = output.reshape(-1, 1)
            #calculate jacobian matrix for output
            jacobian = np.diagflat(output) - np.dot(output, output.T)
            #calculate sample-wise gradient and add to array
            self.dinputs[index] = np.dot(jacobian, dvalue)

class Cost:
    def calculate(self, output, y):
        sample_costs = self.forward(output, y)
        batch_cost = np.mean(sample_costs)
        return batch_cost

#Categorical Cross Entropy loss
    #handles scalar class values or one-hot encoding
class CrossEntropyCost(Cost):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #avoid inf values
        if len(y_true.shape) == 1: # if scalar values vector
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else: # if one-hot encoding matrix
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        neg_log_likelihood = -np.log(correct_confidences)
        return neg_log_likelihood
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues) #num samples
        labels = len(dvalues[0])
        #if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        #calculate gradient
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples #normalize graident

#combined softmax activation and cross-entropy cost for faster backward step
class SoftmaxCostActivation():
    #create activation and cost function objects
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.cost = CrossEntropyCost()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs) #activation function for output layer
        self.output = self.activation.output
        return self.cost.calculate(self.output, y_true)
  
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #change to one-hot if needed
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1 #calculate gradient
        self.dinputs = self.dinputs / samples #normalize

#Stochastic gradient descent optimizer
class SGDOptimizer:
    def __init__(self, learningrate=1, decay=0):
        self.learningrate = learningrate
        self.current_learningrate = learningrate
        self.decay = decay
        self.iterations = 0

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learningrate = self.learningrate * (1. / (1. + self.decay * self.iterations))
    
    #params minus learning rate * gradient
    def update_params(self, layer):
        layer.weights += -self.learningrate*layer.dweights
        layer.biases += -self.learningrate*layer.dbiases

    #iteration count
    def count(self):
        self.iterations += 1

#========================MAIN===============================#
#create custom dataset
X, y = create_spiral_data(100, 3) #100 samples, 3 classes
#plot_data(X)

#construct the neural net

layer0 = Layer(2,32) #initalize first layer -- 2 inputs, 64 nuerons (outputs)
l0_activation = ReLUActivation() #define activation function for layer1

layer1 = Layer(32,32) #initalize first layer -- 2 inputs, 64 nuerons (outputs)
l1_activation = ReLUActivation() #define activation function for layer1

layer2 = Layer(32,3) #define output layer -- 64 in, 3 out
l2_activation = SoftmaxCostActivation() #combined cost and softmax activation function

#initialize optimizer with a learning rate decay 
optimizer = SGDOptimizer(decay=1e-3)

#cycle through multiple times
for epoch in range(10001):

    #pass data through the model
    layer0.forward(X)
    l0_activation.forward(layer0.output)
    layer1.forward(l0_activation.output) #pass through layer1
    l1_activation.forward(layer1.output)
    layer2.forward(l1_activation.output) #pass through layer2 (output layer)

    #calculate cost -- pass through output layer activation function and cost function
    cost = l2_activation.forward(layer2.output, y)

    #calculate accuracy
    predictions = np.argmax(l2_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    #print("loss:", cost)
    #print("acc:", accuracy)

    if (epoch%1000 == 0):
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {cost:.3f}, ' +
              f'lr: {optimizer.current_learningrate}')

    #backpropagation
    l2_activation.backward(l2_activation.output, y)
    layer2.backward(l2_activation.dinputs)
    l1_activation.backward(layer2.dinputs)
    layer1.backward(l1_activation.dinputs)
    l0_activation.backward(layer1.dinputs)
    layer0.backward(l0_activation.dinputs)

    #optimize (update weights and biases for each layer)
    optimizer.pre_update_params()
    optimizer.update_params(layer0)
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.count()

plot_results(predictions)
