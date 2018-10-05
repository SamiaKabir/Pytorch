import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x
    


### Define the input and the target

import numpy as np 
# import tensorflow as tf

arr1 = np.array([[0, 0],[0,1],[1,0],[1,1]])
# x=tf.convert_to_tensor(arr1)
x=Variable(torch.FloatTensor(arr1))
# sess = tf.Session()

print(x)
 
arr2 = np.array([[0],[1],[1],[0]])
# y=tf.convert_to_tensor(arr2)
y=Variable(torch.FloatTensor(arr2))
print(y)

### Create an instance of the Net class
net = Net()
print(net)


### Define the loss and create your optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)


### Main training loop
for i in range(5000):
    
 # Forward pass: Compute predicted y by passing x to the model
    y_pred= net(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if(i%50==0):
       print(i, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### Testing the network
print ("Finished training...")
print(y)
print (torch.round(net(x)))
print(list(net.parameters()))