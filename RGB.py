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

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3,8,3,2,1),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(8,16,3,2,1),
            nn.ReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(16,32,3,2,1),
            nn.ReLU(),
        )   

        self.encoder4 = nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.ReLU(),
        )   

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32,16,4,2,1),
            nn.ReLU(),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.ReLU(),
        )

        self.decoder4= nn.Sequential(
            nn.ConvTranspose2d(8,3,4,2,1),
            nn.Sigmoid(),
        )
        


    def forward(self, x):
        encoded1= self.encoder1(x)
        encoded2= self.encoder2(encoded1)
        encoded3= self.encoder3(encoded2)
        encoded4= self.encoder4(encoded3)
        
        decoded1= self.decoder1(encoded4)
        decoded1= decoded1+encoded3

        decoded2= self.decoder2(decoded1)
        decoded2= decoded2+encoded2

        decoded3= self.decoder3(decoded2)
        decoded3= decoded3+encoded1


        decoded4 = self.decoder4(decoded3)
        # out = decoded+x
        out= decoded4
        return out

### Create an instance of the Net class
net = Net()

## Loading the training and test sets

# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
transform = transforms.ToTensor()

trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)




### Define the loss and create your optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()


### Testing the network on 1st 100 test images and computing the loss

def testing():
    loss=0
    with torch.no_grad():
        for i,data in enumerate(testloader,0):
            input, dummy = data
            target = input

            # preparing the data by downsampling and then upsampling
            image1=torch.nn.functional.interpolate(input,size=48,mode='bilinear')
            input=torch.nn.functional.interpolate(image1,size=96,mode='bilinear')
        
            output=net(input) 
            loss = loss_func(output, target)      # mean square error
            if i==19 :
                break

        print("Loss in Test Results....")
        print(loss)





### Main training loop
for epoch in range(5):
    
    for i, data in enumerate(trainloader, 0):
        
		## Getting the input and the target from the training set
        image, dummy = data
        target = image
        # preparing the data by downsampling and then upsampling
        image1=torch.nn.functional.interpolate(image,size=48,mode='bilinear')
        input=torch.nn.functional.interpolate(image1,size=96,mode='bilinear')

        pred= net(input)
        # print(i)
        loss = loss_func(pred, target)      # mean square error
        if i%500==0 :
              testing()


        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        



### Displaying or saving the results as well as the ground truth images for the first five images in the test set

#show image
def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

for i, data in enumerate(testloader, 0):
        input, dummy = data
        target = input

        # preparing the data by downsampling and then upsampling
        image1=torch.nn.functional.interpolate(input,size=48,mode='bilinear')
        input=torch.nn.functional.interpolate(image1,size=96,mode='bilinear')
        output= net(input) 
       # get some random training images
        images  = output
        # print("input...")
        # print(input)
        # print("output...")
        # print(output)
       # show images
        imshow(torchvision.utils.make_grid(images))
        if(i==5):
            break
        # print labels
        print(' '.join())
