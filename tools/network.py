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
                    assert layers[i].outputSize == layers[i + 1].inputSize, f"Input/Output values do not match for layer {i}"
                self.layerList.append(layers[i])
        except Exception as error:
            print("Error: " + str(error))
        
    def fitNetwork(self):
        pass

    def calculateOutput(self, inputVals):
        try:
            assert len(inputVals) == self.inputSize, "Invalid Input Size"
            for x in inputVals:
                assert type(x) == float or type(x) == int, f"Invalid Input Type: {x}"

            for layer in self.layers:
                inputVals = layer.run(inputVals) #Passes input through layer and returns output as new input

            return(inputVals) #Output

        except Exception as error:
            print("Error: " + str(error))

    def printNetwork(self):
        for layer in self.layerList:
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

        #Random Init of weights and bias
        ##Bias
        for i in range(outputSize): #Output
                self.bias.append(random.uniform(-1,1)) #Append random value of bias, 1 bias value for each output

        ##Weights
        for j in range(inputSize): #Rows
            rowTemp = []
            for i in range(outputSize): #Columns
                rowTemp.append(random.uniform(-1,1)) #Append a random floar between -1 and 1
            
            self.weights.append(rowTemp) #Append the entire rows weights to the full weight matrix

    def printWeights(self):
        print("Weights")
        print(self.weights)
        print("Bias")
        print(self.bias)