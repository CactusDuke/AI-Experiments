from network import *

test = Network()
test.addLayers([
    Linear(10, 7),
    ReLU(),
    Linear(7, 4),
    ReLU()
])

#print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#test.printNetwork()

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(test.calculateOutput([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
