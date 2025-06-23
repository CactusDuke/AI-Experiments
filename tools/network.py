import random
import numpy as np
import math

class Network:
    def __init__(self):
        self.numLayers = 0
        self.inputSize = 0
        self.layerList = []

    def addLayers(self, layers):
        try:
            self.inputSize = layers[0].inputSize
            for i in range(len(layers)):
                if i < (len(layers) - 1):
                    if layers[i + 1].type == 1:
                        layers[i + 1].inputSize = layers[i].outputSize
                        layers[i + 1].outputSize = layers[i].outputSize

                    assert layers[i].outputSize == layers[i + 1].inputSize, f"Input/Output values do not match for layer {i}"
                self.layerList.append(layers[i])
        except Exception as error:
            print("Error: " + str(error))
        
    def fitNetwork(self):
        pass

    def calculateOutput(self, inputV):
        try:
            assert len(inputV) == self.inputSize, "Invalid Input Size"
            for x in inputV:
                assert type(x) == float or type(x) == int, f"Invalid Input Type: {x}"

            for layer in self.layerList:
                inputV = layer.run(inputV) #Passes input through layer and returns output as new input


            return(inputV) #Output

        except Exception as error:
            print("Error: " + str(error))

    def printNetwork(self):
        for layer in self.layerList:
            if layer.type == 0:
                layer.printWeights()

class Linear:
    def __init__(self, inputSize, outputSize):
        assert type(inputSize) == int, "Invalid type for input size"
        assert inputSize > 0, "Input must be greater than 0"
        assert type(outputSize) == int, "Invalid type for output size"
        assert outputSize > 0, "Output must be greater than 0"


        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weights = []
        self.bias = []
        self.type = 0 #Determines if it is a layer, an activation function, etc.

        #Random Init of weights and bias
        ##Bias
        temp = []
        for i in range(outputSize): #Output
            temp.append(random.uniform(-1,1)) #Append random value of bias, 1 bias value for each output
        self.bias.append(temp) #It is done like this to ensure that bias is a matrix and not 1d

        ##Weights
        for j in range(inputSize): #Rows
            rowTemp = []
            for i in range(outputSize): #Columns
                rowTemp.append(random.uniform(-1,1)) #Append a random floar between -1 and 1
            
            self.weights.append(rowTemp) #Append the entire rows weights to the full weight matrix

    def run(self, inputV):
        inputVals = []
        inputVals.append(inputV) #Although inputed as a 1d array it needs to be an (inputsize) X 1 matrix
        output = np.matmul(inputVals, self.weights) #Multiply weights by the values
        output = np.add(output, self.bias) #Add the biases
        return(output) #Return the output


    def printWeights(self):
        print("Weights")
        print(self.weights)
        print("Bias")
        print(self.bias)

class ReLU:
    def __init__(self):
        self.inputSize = 0
        self.outputSize = 0
        self.type = 1 #Determines if it is a layer, an activation function, etc
        
    def run(self, inputV):
        output = []
        for i in inputV[0]:
            output.append(float(max(0, i))) #Max between 0 and the value

        return(output)


class TANH:
    def __init__(self):
        self.inputSize = 0
        self.outputSize = 0
        self.type = 1 #Determines if it is a layer, an activation function, etc

    def run(self, inputV):
        output = []
        for i in inputV[0]:
            output.append(math.tanh(i)) #tanh of the value
        return(output)

class Sigmoid:
    def __init__(self):
        self.inputSize = 0
        self.outputSize = 0
        self.type = 1 #Determines if it is a layer, an activation function, etc
        
    def run(self, inputV):
        output = []
        for i in inputV[0]:
            output.append(1 / (1 + math.exp(-1 * i))) #Sigmoid of the valuse
        return(output)

