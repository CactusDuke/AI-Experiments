from network import *


test = Network()
test.addLayers([
    Con2D(3, 3, [[1, 2]], 1),
    Flatten(),
    Linear(6, 9),
    SoftMax()
])




print(test.calculateOutput([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


"""
test = Network()
test.addLayers([
    Linear(10, 7),
    ReLU(),
    Linear(7, 4),
    ReLU()
])

#Calculation
startXcord = 0 #Top left value of kernal for each itertion. Used to move 
startYcord = 0
inputV = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

#Add Padding
padding = 2
rowPad = [0] * (3 + (2 * padding)) #Empty row to append to the input to add padding
for row in inputV:
    for p in range(padding):
        row.insert(0,0) #Insert 0 into start of list
        #np.insert(row, 0, 0)
        row.append(0) #append 0 to end of list
        #np.append(row, 0)
for p in range(padding): 
    inputV.insert(0,rowPad) #Insert row padding into start of list
    #np.insert(inputV, 0, rowPad)
    inputV.append(rowPad) #append row padding to end of list
    #np.append(inputV, rowPad)

inputV = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
inputV = np.array(inputV)

#print(inputV)
tempMatrix = inputV.copy()
#print(tempMatrix[1:3, 0:3])
#print(inputV)
#print(tempMatrix[rowStart:rowEnd, colStart:colEnd])

kernal = [[1, 2], [3, 4]]
startYcord=0
startXcord=0
#tempMatrix[startYcord:(startYcord + len(kernal[0])), startXcord:(startXcord + len(kernal))]
output = []
for j in range(2):
    startXcord=0
    row = []
    for i in range(2):
        tempMatrix = inputV[startYcord:(startYcord + len(kernal[0])), startXcord:(startXcord + len(kernal))]
        startXcord += 1
        #Fuckin dot product gets funky with 2d arrays
        dotProd = 0
        for i in range(len(kernal)):
            dotProd += np.dot(tempMatrix[i], kernal[i]) #Splits the 2d arrays into 1d ones and combines dot values. [[1, 2], [3, 4]] dot [[1, 1], [2, 2]] = 1*1 + 2*1 + 3*2 + 4*2 = [1, 2] dot [1, 1] + [3, 4] dot [2, 2]
        row.append(int(dotProd)) #Append to row
    output.append(list(row))
    startYcord += 1
print(output)
"""
