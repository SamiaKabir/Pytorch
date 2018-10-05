import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,4,3,2,1),
            nn.ReLU(),
            nn.Conv2d(4,8,3,2,1),
            nn.ReLU(),
            nn.Conv2d(8,16,3,2,1),
            nn.ReLU(),
        )   

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,4,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(4,3,4,2,1),
            # nn.Linear(38*38,32*32),
        )


    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded.size())
        decoded = self.decoder(encoded)
        # print(decoded.size())
        return decoded

### Create an instance of the Net class
net = Net()

## Loading the training and test sets

# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)



### Define the loss and create your optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

### Main training loop
for epoch in range(2):
    
    for i, data in enumerate(trainloader, 0):
        
		## Getting the input and the target from the training set
        input, dummy = data
        target = input

        pred= net(input)

        loss = loss_func(pred, target)      # mean square error
        if i%10000==0 :
           print(loss.item())


        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        

### Testing the network on 10,000 test images and computing the loss
loss=0
with torch.no_grad():
    for data in testloader:
        input, dummy = data
        target = input

        output=net(input) 
        loss = loss_func(output, target)      # mean square error


print("Loss in Test Results....")
print(loss)

### Displaying or saving the results as well as the ground truth images for the first five images in the test set

#show image
def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

for i, data in enumerate(testloader, 0):
        input, dummy = data
        target = input
        output= net(input) 
       # get some random training images
        images  = output
        # print("input...")
        # print(input)
        # print("output...")
        # print(output)
       # show images
        # imshow(torchvision.utils.make_grid(input))
        imshow(torchvision.utils.make_grid(images))
        plt.savefig(images)
        if(i==5):
            break
        # print labels
        print(' '.join())

