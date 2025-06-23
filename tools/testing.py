from network import *

test = Network()
test.addLayers([
    Linear(3, 1),
    Linear(1, 1)
])

test.printNetwork()