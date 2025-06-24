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
            if layers[0].type == 2: #If start is 1d
                self.inputSize = layers[0].inputSize
            elif layers[0].type == 3: #If start is 2d
                self.inputSize = layers[0].inputSizeY #Number of rowsa
            for i in range(len(layers)):
                if i < (len(layers) - 1):
                    if layers[i + 1].type == 0: #Base Layer Functions N -> N
                        assert layers[i].outputSize != 0, f"Incorrect Output Format for layers {i}" #Checks that the correct output type exists
                        assert layers[i].outputSize == layers[i + 1].inputSize, f"Input/Output values do not match for layer {i}"

                    elif layers[i + 1].type == 1: #Activation functions N
                        assert layers[i].outputSize != 0, f"Incorrect Output Format for layers {i}" #Checks that the correct output type exists
                        layers[i + 1].inputSize = layers[i].outputSize
                        layers[i + 1].outputSize = layers[i].outputSize
                    
                    elif layers[i + 1].type == 2: #Functions like Flatten. 2d -> 1d
                        assert layers[i].outputSizeX != 0, f"Incorrect Output Format for layers {i}" #Checks that the correct output type exists
                        layers[i + 1].inputSizeX = layers[i].outputSizeX
                        layers[i + 1].inputSizeY = layers[i].outputSizeY
                        layers[i + 1].outputSize = layers[i].outputSizeY * layers[i].outputSizeX

                    elif layers[i + 1].type == 3: #2d Functions R2 -> R2
                        assert layers[i].outputSizeX != 0, f"Incorrect Output Format for layers {i}" #Checks that the correct output type exists
                        assert layers[i].outputSizeX == layers[i + 1].inputSizeX, f"Input/Output X values do not match for layer {i}"
                        assert layers[i].outputSizeY == layers[i + 1].inputSizeY, f"Input/Output Y values do not match for layer {i}"

                self.layerList.append(layers[i])
        except Exception as error:
            print("Error: " + str(error))

    def calculateOutput(self, inputV):
        try:
            assert len(inputV) == self.inputSize, "Invalid Input Size"
            for x in inputV:
                assert type(x) != str, f"Invalid Input Type: {x}"

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
        return(output[0]) #Return the output


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

class LogSoftMax:
    def __init__(self):
        self.inputSize = 0
        self.outputSize = 0
        self.type = 1 #Determines if it is a layer, an activation function, etc
        
    def run(self, inputV):
        output = []
        total = 0
        for i in inputV:
            total += math.exp(i) #Adds value to total

        for i in inputV:
            output.append(float(math.log(math.exp(i)/total))) #Log of the exponent percentage
        return(output)

class SoftMax:
    def __init__(self):
        self.inputSize = 0
        self.outputSize = 0
        self.type = 1 #Determines if it is a layer, an activation function, etc
        
    def run(self, inputV):
        output = []
        total = 0
        for i in inputV:
            total += math.exp(i) #Adds value to total

        for i in inputV:
            output.append(float(math.exp(i) / total)) #Log of the exponent percentage
        return(output)

class Flatten:
    def __init__(self):
        self.inputSizeX = 0
        self.inputSizeY = 0
        self.outputSize = 0
        self.type = 2 #Determines if it is a layer, an activation function, etc
        
    def run(self, inputV):
        output = []
        for row in inputV:
            for value in row:
                output.append(value) #Flattens 2d matric into 1d

        return(output)

class Con2D:
    def __init__(self, inputSizeX, inputSizeY, kernal, stride, padding = 0):
        assert len(kernal[0]) > 0, "Kernal must be 2d array"
        assert len(kernal) > 0, "Kernal cannot be empty"


        self.inputSizeX = inputSizeX
        self.inputSizeY = inputSizeY
        self.kernal = np.array(kernal)
        self.stride = stride
        self.padding = padding
        self.type = 3 #Determines if it is a layer, an activation function, etc.

        self.outputSizeX = int(((inputSizeX - len(kernal[0]) + (2 * padding)) / stride) + 1)
        self.outputSizeY = int(((inputSizeY - len(kernal) + (2 * padding)) / stride) + 1)

    
    def run(self, inputV):      
        #Add Padding
        rowPad = [0] * (self.inputSizeX + (2 * self.padding)) #Empty row to append to the input to add padding
        for row in inputV:
            for p in range(self.padding):
                row.insert(0,0) #Insert 0 into start of list
                row.append(0) #append 0 to end of list
        for p in range(self.padding): 
            inputV.insert(0,rowPad) #Insert row padding into start of list
            inputV.append(rowPad) #append row padding to end of list

        #Make list into array
        inputV = np.array(inputV) #Makes it into a numpy array. Allows for easier matrix splitting


        #Calculation
        startXcord = 0 #Top left value of kernal for each itertion. Used to move around
        startYcord = 0
        output = []
        #tempMatrix[rowStart:rowEnd, colStart:colEnd]. Here to show what each value does
        
        for j in range(self.outputSizeY):
            startXcord = 0
            row = []
            for i in range(self.outputSizeX):
                #tempMatrix = inputV[startYcord:(startYcord + len(self.kernal[0])), startXcord:(startXcord + len(self.kernal))] #Temp matrix with the subset of the input that is being studied
                tempMatrix = inputV[startYcord:(startYcord + len(self.kernal)), startXcord:(startXcord + len(self.kernal[0]))] #Temp matrix with the subset of the input that is being studied
                #Fuckin dot product gets funky with 2d arrays
                dotProd = 0
                for i in range(len(self.kernal)):
                    dotProd += np.dot(tempMatrix[i], self.kernal[i]) #Splits the 2d arrays into 1d ones and combines dot values. [[1, 2], [3, 4]] dot [[1, 1], [2, 2]] = 1*1 + 2*1 + 3*2 + 4*2 = [1, 2] dot [1, 1] + [3, 4] dot [2, 2]
                row.append(float(dotProd)) #Append to row
                startXcord += self.stride #Move to next start point

            output.append(list(row))
            startYcord += self.stride
        return(output)